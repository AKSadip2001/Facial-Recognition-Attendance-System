from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime

import csv

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    now = datetime.now()
    current_date = now.strftime("%d-%m-%Y")
    f = open(current_date+'.csv')
    with f as file:
        reader = csv.reader(file)
        next(reader)
        data={
            "csv": reader,
            "date": current_date
        }
        return templates.TemplateResponse("index.html", {"request": request, "data": data})
