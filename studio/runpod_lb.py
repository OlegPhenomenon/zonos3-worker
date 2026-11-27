"""RunPod Serverless load-balancing mode of the studio-prep worker (the fallback when Vast has no worker): a plain HTTP server
on PORT (default 80) that RunPod routes https://<endpoint id>.api.runpod.ai/<path> to, the API key checked by RunPod in front.
The routes and their answers are vast_worker.py's, the payload is the request body and the answer the response body (no
{"result": ...} wrapping; a failure is still an answer {"error": "<code>", ...}, HTTP 200). GET /ping answers 204 while the
separator and the emotion model load, 200 once they are ready (only 200 gets traffic).

The stems are on this worker's disk, as on Vast: the fallback endpoint runs one worker (max 1), so every request of a job
reaches the worker that separated its audio; a new worker answers not_prepared and the runner prepares again.
"""
import logging, os, threading

from aiohttp import web

from . import vast_worker as vw

log = logging.getLogger('studio.runpod')
ready = threading.Event()
ROUTES = {'/prepare': vw.prepare, '/ser': vw.ser, '/mix_plan': vw.mix_plan, '/mix_piece': vw.mix_piece, '/mix_cue': vw.mix_cue}


async def ping(_request):
    return web.Response(status=200 if ready.is_set() else 204)


def handler(fn):
    async def serve(request):
        if not ready.is_set():
            return web.json_response({'error': 'loading'}, status=503)
        try:
            payload = await request.json()
        except ValueError:
            return web.json_response({'error': 'bad_request', 'detail': 'body is not JSON'})
        if not isinstance(payload, dict):
            return web.json_response({'error': 'bad_request', 'detail': 'body is not an object'})
        return web.json_response(await fn(**payload))
    return serve


def load():
    try:
        vw.load_separator()
    except Exception:  # noqa: BLE001 - a worker stuck at 204 would be billed while it never serves: exit, RunPod marks it unhealthy
        log.exception('studio models did not load')
        os._exit(1)
    ready.set()


def main():
    logging.basicConfig(level=logging.INFO)
    threading.Thread(target=load, daemon=True).start()
    app = web.Application(client_max_size=30 * 1024 * 1024)  # RunPod's own request limit
    app.add_routes([web.get('/ping', ping)] + [web.post(path, handler(fn)) for path, fn in ROUTES.items()])
    web.run_app(app, host='0.0.0.0', port=int(os.environ.get('PORT', '80')))


if __name__ == '__main__':
    main()
