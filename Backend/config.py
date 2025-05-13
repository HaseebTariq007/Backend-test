import os

class Config:
    # MySQL Database Configurations according to my own local phpmyadmin databases
    MYSQL_HOST = 'localhost'  # Your DB is running locally
    MYSQL_USER = 'root'        # Default MySQL user
    MYSQL_PASSWORD = ''        # Set your own MySQL password here if you have one
    MYSQL_DB = 'child_profiles_db'  # This is the DB you just created

    # Max upload size (no need to change unless needed)
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit
    
    # Folder where uploaded images could be stored (if needed separately)
    UPLOAD_FOLDER = 'uploads'
    
    # Flask Secret Key (change for security)
    SECRET_KEY = 'Chunno5july'  # Replace with a random strong key in production
