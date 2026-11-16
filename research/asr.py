"""Transcribe Zonos takes with Gemini as a listener: how were the Latin words actually pronounced?
Usage: GEMINI_API_KEY=... python3 asr.py a.wav b.wav ..."""
import base64, json, os, sys, urllib.request

PROMPT = ('Transcribe this Russian speech exactly as pronounced. Write every word in Cyrillic as it sounds. '
          'If a word is spelled out letter by letter, write the letter names separated by hyphens (e.g. "пэ-игрек-тэ"). '
          'Then on a new line starting with "NOTES:" say, for each English term you heard, whether it was spoken as '
          'one word or spelled letter by letter.')
key = os.environ['GEMINI_API_KEY']
for path in sys.argv[1:]:
    body = {'contents': [{'parts': [{'text': PROMPT}, {'inline_data': {'mime_type': 'audio/wav',
            'data': base64.b64encode(open(path, 'rb').read()).decode()}}]}], 'generationConfig': {'temperature': 0}}
    req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={key}',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    print(f'== {os.path.basename(path)}\n{out["candidates"][0]["content"]["parts"][0]["text"].strip()}\n')
