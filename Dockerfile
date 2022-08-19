FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run"]

CMD ["src/app.py"]

EXPOSE 8501