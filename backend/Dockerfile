FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY="your_api_key"

WORKDIR /src
COPY api ./

RUN pip install --upgrade pip
RUN pip install fastapi
RUN pip install "uvicorn[standard]"
RUN pip install requests
RUN pip install langchain
RUN pip install langchain_openai
RUN pip install langgraph
RUN pip install -U langmem
RUN pip install numpy
RUN pip install langchain_community

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]
