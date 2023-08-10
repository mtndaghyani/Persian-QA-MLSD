# Define a function to measure latency
import time
import wandb

KEY = 'b47b329c13afbaed9c77be312a4eaea3dcbbdfd1'
PROJECT = 'Persian Question Answering'
NAME = 'Model latency'


def measure_latency(model):
    question_mock = 'انیشتین در چه سالی به دنیا آمد؟'  # Example input data
    context_mock = 'انیشتین در سال 1910 به دنیا آمده است.'  # Example input data
    start_time = time.time()
    _ = model(question_mock, context_mock)
    latency = time.time() - start_time
    return latency


def run_monitoring(model):
    wandb.login(key=KEY)
    wandb.init(
        project=PROJECT,
        name=NAME,
    )

    while True:
        latency = measure_latency(model)
        wandb.log({'Latency': latency})
        time.sleep(20)
