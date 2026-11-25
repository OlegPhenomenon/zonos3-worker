"""Location library: recognise that a scene returns to an earlier location and give all its visits one treatment.

Port of the lab's `location_library.py`. channel matching measures the channel of every acoustic scene on its own; a location that
is cut away from and returned to is then dubbed with as many slightly different rooms / compressors as it has visits, and short
visits (1-10 s of speech) carry too little evidence for a trustworthy fit. Netflix asks for one consistent room per place.

Method (all numbers come from the clip itself)
1. Every scene has a spectral signature (scenes.detect: speaker-normalised long-term speech spectrum in 1/3-octave bands) and a
   measurement noise sigma that shrinks with 1/sqrt(speech seconds).
2. Pair distance z_ij = rms-dB distance of the signatures / sqrt(noise_i^2 + noise_j^2): how many measurement sigmas apart two
   scenes are. z < LOC_Z means they cannot be told apart by their audio.
3. Complete-linkage agglomeration (a scene joins a cluster only when it is indistinguishable from EVERY member, no chaining), stopped
   when the closest pair has z >= LOC_Z. Scenes with less than MIN_SPEECH_S of speech neither bridge nor create a cluster: they are
   attached to the nearest cluster in dB only when z < LOC_Z, otherwise they stay alone.
4. Each location gets `weights` = speech seconds of each member (the evidence): the consumer averages the location's tone with them
   and takes the room / dynamics of the member with the largest weight (`rep_scene`).

API
    cluster(scene_doc, chan) -> dict
        {"z_threshold", "min_speech_s", "locations": [{"id", "scenes": [scene ids], "weights": {scene id str: speech s}, "rep_scene",
         "speech_s"}], "scene_to_location": {scene id str: location id}, "pair_z": {"i-j": z}}   (lab locations.json)
        scene_doc = scenes.detect(...) document (scenes with id, speech_s, signature, signature_noise); chan = channel document;
        the two must list the same scenes (same start / end), otherwise RuntimeError.
"""
import numpy as np

LOC_Z = 2.0  # two scenes are the same place when their signatures differ by < 2 measurement sigmas (95 % two-sided, no tuning)
MIN_SPEECH_S = 5.0  # below this a scene cannot found or bridge a cluster (its noise sigma is > 2x that of a 20 s scene)


def signature_distance(a, b):
    """rms dB difference of two signatures."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d * d)))


def pair_z(scenes):
    """(z, dist) matrices (n, n): measurement sigmas and dB between every pair of scenes; inf where a signature is missing."""
    n = len(scenes)
    z = np.full((n, n), np.inf)
    dist = np.full((n, n), np.inf)
    sig = [s["signature"] for s in scenes]
    noise = np.array([s["signature_noise"] if s["signature"] is not None else np.nan for s in scenes], float)
    for i in range(n):
        z[i, i], dist[i, i] = 0.0, 0.0
        for j in range(n):
            if i != j and sig[i] is not None and sig[j] is not None:
                d = signature_distance(sig[i], sig[j])
                dist[i, j] = d
                z[i, j] = d / np.sqrt(noise[i] ** 2 + noise[j] ** 2)
    return z, dist


def agglomerate(scenes, z, dist):
    n = len(scenes)
    speech = [s["speech_s"] for s in scenes]
    big = [i for i in range(n) if speech[i] >= MIN_SPEECH_S and scenes[i]["signature"] is not None]
    clusters = [[i] for i in big]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                zz = max(z[i, j] for i in clusters[a] for j in clusters[b])
                if zz < LOC_Z and (best is None or zz < best[0]):
                    best = (zz, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] += clusters[b]
        del clusters[b]
    for i in range(n):  # thin scenes: nearest cluster in dB if indistinguishable from it, else alone
        if i in big:
            continue
        cand = [(np.mean([dist[i, j] for j in c]), k) for k, c in enumerate(clusters) if scenes[i]["signature"] is not None
                and max(z[i, j] for j in c) < LOC_Z]
        if cand:
            clusters[min(cand)[1]].append(i)
        else:
            clusters.append([i])
    return clusters


def cluster(scene_doc, chan):
    """See the module docstring."""
    scenes = scene_doc["scenes"]
    if [(round(s["start"], 2), round(s["end"], 2)) for s in scenes] != [(round(c["start"], 2), round(c["end"], 2)) for c in chan["scenes"]]:
        raise RuntimeError("scene detection does not match the channel document")
    z, dist = pair_z(scenes)
    clusters = agglomerate(scenes, z, dist)
    locs = []
    for k, c in enumerate(sorted(clusters, key=lambda c: -sum(scenes[i]["speech_s"] for i in c))):
        ids = sorted(scenes[i]["id"] for i in c)
        w = {str(scenes[i]["id"]): round(float(scenes[i]["speech_s"]), 1) for i in c}
        locs.append({"id": k, "scenes": ids, "weights": w, "rep_scene": int(max(c, key=lambda i: scenes[i]["speech_s"])),
                     "speech_s": round(sum(w.values()), 1)})
    return {"z_threshold": LOC_Z, "min_speech_s": MIN_SPEECH_S, "locations": locs,
            "scene_to_location": {str(s): L["id"] for L in locs for s in L["scenes"]},
            "pair_z": {f"{scenes[i]['id']}-{scenes[j]['id']}": round(float(z[i, j]), 2) for i in range(len(scenes)) for j in range(i + 1, len(scenes))
                       if np.isfinite(z[i, j])}}
