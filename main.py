import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from predictor import predictor

# Setting Up server
app = FastAPI()


# Request body data model
class Info(BaseModel):
    context: str
    question: str


def post_process(output):
    return output[0]['text'].strip()


def preprocess(question, context):
    # Apply required preprocessing to user inputs
    return question, context


# Path Operation Functions
@app.get('/alive')
def test_liveness():
    return {"message": "Hello! The server is alive :D"}


@app.post("/ask")
def predict(request: Info):
    context = request.context
    question = request.question
    question, context = preprocess(question, context)
    output = predictor(question, context, batch_size=1)
    print(output)
    answer = post_process(output)
    return {"answer": answer}


def run_app():
    # TODO:
    # uvicorn.run(app, host="0.0.0.0", port=8000) Uncomment for deployment
    uvicorn.run(app, host="127.0.0.1", port=8000)  # Comment this line after deployment


if __name__ == "__main__":
    # t1 = threading.Thread(target=run_app)
    # t2 = threading.Thread(target=run_monitoring, args=[model, ])
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    run_app()
