import uuid

import runpod

import common

boot_id = str(uuid.uuid4())
common.start_server()


def handler(job):
    return {**common.generate(job['input']), 'boot_id': boot_id}


runpod.serverless.start({'handler': handler})
