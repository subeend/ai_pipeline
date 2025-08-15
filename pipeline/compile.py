
import os
from kfp import compiler
from pipeline import yolo_pipeline

PIPE_JSON = os.environ.get("PIPE_JSON", "yolo_compiled.json")
compiler.Compiler().compile(yolo_pipeline, PIPE_JSON)
print("compiled:", PIPE_JSON)


