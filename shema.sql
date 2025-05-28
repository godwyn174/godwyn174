CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL
);

CREATE TABLE IF NOT EXISTS fields (
    field_id SERIAL PRIMARY KEY,
    field_name VARCHAR(100),
    gps_coordinates VARCHAR
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id SERIAL PRIMARY KEY,
    field_id INT REFERENCES fields(field_id),
    collection_date DATE NOT NULL,
    ph FLOAT,
    organic_matter FLOAT,
    nitrogen FLOAT,
    phosphorus FLOAT,
    potassium FLOAT,
    moisture FLOAT
);
from app import db, create_app
app = create_app()
with app.app_context():
    db.create_all()