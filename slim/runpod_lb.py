"""RunPod Serverless load-balancing mode (the fallback when Vast has no worker): a plain HTTP server on PORT (default 80)
that RunPod routes https://<endpoint id>.api.runpod.ai/<path> to, the API key checked by RunPod in front.

  GET  /ping  204 while zonos3-server loads, 200 once it answers (RunPod's health check: only 200 gets traffic)
  POST /tts   the Vast payload as the body -> the same answer as the Vast route's result (common.generate), JSON;
              a failure is HTTP 500 {"error": "..."}
"""
import asyncio, logging, os, threading

from aiohttp import web

import common

log = logging.getLogger('zonos.runpod')
ready = threading.Event()


async def ping(_request):
    return web.Response(status=200 if ready.is_set() else 204)


async def tts(request):
    if not ready.is_set():
        return web.json_response({'error': 'loading'}, status=503)
    payload = await request.json()
    try:
        return web.json_response(await asyncio.get_running_loop().run_in_executor(None, common.generate, payload))
    except Exception as e:  # noqa: BLE001 - the caller retries; the worker keeps serving
        log.exception('tts failed')
        return web.json_response({'error': str(e)[:500]}, status=500)


def load():
    try:
        common.start_server()
    except Exception:  # noqa: BLE001 - a worker stuck at 204 would be billed while it never serves: exit, RunPod marks it unhealthy
        log.exception('zonos3-server did not start')
        os._exit(1)
    ready.set()
    log.info('zonos3-server ready in %.1f s (%s)', common.load_s, common.gpu)


def main():
    logging.basicConfig(level=logging.INFO)
    threading.Thread(target=load, daemon=True).start()
    app = web.Application(client_max_size=30 * 1024 * 1024)  # RunPod's own request limit
    app.add_routes([web.get('/ping', ping), web.post('/tts', tts)])
    web.run_app(app, host='0.0.0.0', port=int(os.environ.get('PORT', '80')))


if __name__ == '__main__':
    main()
