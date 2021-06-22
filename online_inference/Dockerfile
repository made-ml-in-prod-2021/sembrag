FROM python:3.8
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


COPY model/clf_hw2.pkl model/clf_hw2.pkl
COPY src /src
COPY setup.py ./setup.py

RUN pip install -e .

WORKDIR .

ENV PATH_TO_MODEL="/model/clf_hw2.pkl"

WORKDIR src/online_inference/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]