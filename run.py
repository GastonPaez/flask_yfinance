from app import create_app
import os
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT, default=5000"))
