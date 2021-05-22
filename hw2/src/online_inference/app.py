import logging
import os
from typing import List, Union, Optional

import uvicorn
from fastapi import FastAPI


from src.online_inference.DataClasses_ import Data, ModelOut
from src.online_inference.model import Model

app_name = os.getenv('APP_NAME', 'ML_APP')

logger = logging.getLogger(app_name)

model: Optional[Model] = None


app = FastAPI()


@app.get('/')
def main():
    return f'{app_name} entry point'


@app.on_event('startup')
def load_model():
    global model
    model_path = os.getenv('PATH_TO_MODEL', '..\\..\\model\\clf_hw2.pkl')
    if model_path is None:
        err = f'PATH_TO_MODEL {model_path} is None'
        logger.error(err)
        raise RuntimeError(err)

    model = Model(model_path)


@app.get('/health')
def health() -> bool:
    return not (model is None)


@app.get('/predict/', response_model=List[ModelOut])
def predict(request: Data):
    preds = model.make_predict(request.data, request.features, request.indexes)
    print(preds)
    return preds

@app.get('/test/', response_model=ModelOut)
def test(request: ModelOut):
    print(request)
    return request

if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=os.getenv('PORT', 8000))
