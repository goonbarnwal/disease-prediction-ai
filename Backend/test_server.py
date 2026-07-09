# test_server.py
print("=" * 50)
print("🧪 TESTING FLASK SERVER")
print("=" * 50)

# Try to import Flask
try:
    from flask import Flask
    print("✅ Flask imported successfully")
except ImportError:
    print("❌ Flask not installed")
    print("Run: pip install flask")
    exit()

# Create simple app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Server is working!"

@app.route('/api/test')
def test():
    return {"status": "ok", "message": "Test successful"}

if __name__ == '__main__':
    print("\n🚀 Starting test server on http://localhost:5000")
    print("👉 Press Ctrl+C to stop")
    print("=" * 50)
    app.run(debug=True, port=5000, use_reloader=False)