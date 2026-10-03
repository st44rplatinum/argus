"""One module per job type. Each has run(request, progress) -> result dict.

They run inside the worker's subprocess, so heavy imports go inside run().
A job fails by raising; the worker turns that into a failed job.
"""
