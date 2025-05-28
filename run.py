try:
    from app import create_app
    print("Imported create_app successfully")
    app = create_app()
    print("Created app successfully")
    if __name__ == '__main__':
        print("Starting Flask server...")
        app.run(debug=True, host='127.0.0.1', port=5000)
except Exception as e:
    print(f"Error in run.py: {e}")