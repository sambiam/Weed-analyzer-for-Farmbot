import uvicorn

if __name__ == "__main__":
    uvicorn.run("farmbot_vision.web:app", host="0.0.0.0", port=8099, proxy_headers=True)
