"""Build the reference voice bank for every Zonos language: 10 prebuilt Gemini TTS voices (5 female, 5 male) each read a neutral
~15 s text in that language; the clips are cloned by Zonos through speaker embeddings (compute_embeddings.sh).
A voice keeps its name across languages, so a hero sounds like the same person in every dub.

  GCP_CREDENTIALS=sa.json [GEMINI_API_KEY=...] python3 bank/build_bank.py [code ...]   (default: all Zonos languages)

Speech goes through Vertex AI with a service account (no daily request cap); GEMINI_API_KEY is only needed to
translate the reference text for a language missing from bank/texts.json.
Idempotent: existing clips and translations are kept. Output: bank/refs/<code>/<Voice>.wav, bank/texts.json."""
import base64, io, json, os, sys, time, urllib.error, urllib.request, wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import google.auth.transport.requests
from google.oauth2 import service_account

ROOT = Path(__file__).resolve().parent
KEY = os.environ.get('GEMINI_API_KEY', '')
creds = service_account.Credentials.from_service_account_file(
    os.environ['GCP_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
API = 'https://generativelanguage.googleapis.com/v1beta'
TTS_MODEL, TEXT_MODEL = 'gemini-3.8-flash-tts', 'gemini-3.8-flash'
SOURCE = ('Today we will calmly go through a new topic. First, we will see where it came from and why people talk about '
          'it so much. Then we will discuss a few simple examples from everyday life, and at the end you will be able to '
          'check for yourself whether everything has become clear.')
# Zonos server language codes (model/src/zonos3/registries/languages.json) -> language named for the translator.
LANGS = {'en_us': 'English (US)', 'en_gb': 'English (UK)', 'fr_fr': 'French', 'de': 'German', 'es': 'Spanish',
         'it': 'Italian', 'pt_br': 'Portuguese (Brazil)', 'ja': 'Japanese', 'cmn': 'Mandarin Chinese (simplified)',
         'ko': 'Korean', 'ar': 'Arabic', 'bn': 'Bengali', 'bg': 'Bulgarian', 'cs': 'Czech', 'da': 'Danish',
         'nl': 'Dutch', 'fi': 'Finnish', 'ka': 'Georgian', 'el': 'Greek', 'gu': 'Gujarati', 'he': 'Hebrew',
         'hi': 'Hindi', 'hu': 'Hungarian', 'id': 'Indonesian', 'kn': 'Kannada', 'ms': 'Malay', 'ml': 'Malayalam',
         'mr': 'Marathi', 'no': 'Norwegian (Bokmål)', 'or': 'Odia', 'pl': 'Polish', 'pa': 'Punjabi (Gurmukhi)',
         'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sv': 'Swedish', 'tl': 'Tagalog', 'ta': 'Tamil',
         'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese'}
# 5 female + 5 male neutral narrators (Google's voice descriptions); same names in every language.
VOICES = {'Kore': 'F', 'Aoede': 'F', 'Callirrhoe': 'F', 'Despina': 'F', 'Sulafat': 'F',
          'Charon': 'M', 'Iapetus': 'M', 'Algieba': 'M', 'Schedar': 'M', 'Sadaltager': 'M'}
MIN_S, MAX_S = 8.0, 30.0   # a clip outside this is a failed read (cut off or rambling): retried


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={'x-goog-api-key': KEY, 'Content-Type': 'application/json'})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors='replace').replace(KEY, '[REDACTED]')
            if e.code in (429, 500, 503) and attempt < 5:
                time.sleep(10 * (attempt + 1)); continue
            raise RuntimeError(f'HTTP {e.code}: {body[:300]}') from None


def translate(lang):
    if lang.startswith('en_'):
        return SOURCE
    out = post(f'{API}/models/{TEXT_MODEL}:generateContent', {'contents': [{'parts': [{'text':
        f'Translate into natural spoken {LANGS[lang]}, the way a calm native lecturer would say it. Write numbers as '
        f'words, no abbreviations, no quotes, no comments. Output only the translation.\n\n{SOURCE}'}]}]})
    return out['candidates'][0]['content']['parts'][0]['text'].strip()


def vertex_tts(text, voice, style):
    """Vertex AI generateContent with a service account (as talking_photos worker/src/tts/gemini.rs): no daily cap."""
    creds.refresh(google.auth.transport.requests.Request()) if not creds.valid else None
    body = {'contents': [{'role': 'user', 'parts': [{'text': text, 'speechMetadata': {'style': style}}]}],
            'generationConfig': {'responseModalities': ['AUDIO'], 'speechConfig': {'voiceConfig': {'voice': voice}}}}
    url = (f'https://aiplatform.googleapis.com/v1/projects/{creds.project_id}/locations/global/publishers/google/'
           f'models/{TTS_MODEL}:generateContent')
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 5:
                time.sleep(10 * (attempt + 1)); continue
            raise RuntimeError(f'HTTP {e.code}: {e.read().decode(errors="replace")[:300]}') from None
    part = next(p['inlineData'] for p in reversed(data['candidates'][0]['content']['parts']) if 'inlineData' in p)
    pcm = base64.b64decode(part['data'])
    if pcm[:4] == b'RIFF':
        return pcm
    rate = int(part.get('mimeType', '').partition('rate=')[2] or 24000)   # audio/L16;rate=24000: raw 16-bit mono
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return out.getvalue()


def speak(lang, voice, text):
    target = ROOT / 'refs' / lang / f'{voice}.wav'
    if target.exists():
        return f'{lang}/{voice} kept'
    style = f'Clean studio recording. Natural native {LANGS[lang]} speech, steady pace, neutral delivery, no emotion.'
    for attempt in range(3):
        wav = vertex_tts(text, voice, style)
        with wave.open(io.BytesIO(wav)) as w:
            seconds = w.getnframes() / w.getframerate()
        if MIN_S <= seconds <= MAX_S:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wav)
            return f'{lang}/{voice} {seconds:.1f}s'
    return f'{lang}/{voice} FAILED'


def main(langs):
    texts_path = ROOT / 'texts.json'
    texts = json.loads(texts_path.read_text()) if texts_path.exists() else {}
    for lang in langs:
        if lang not in texts:
            texts[lang] = translate(lang)
            texts_path.write_text(json.dumps(texts, ensure_ascii=False, indent=1))
    jobs = [(lang, v, texts[lang]) for lang in langs for v in VOICES]
    with ThreadPoolExecutor(6) as ex:
        for line in ex.map(lambda j: speak(*j), jobs):
            print(line, flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(LANGS))
