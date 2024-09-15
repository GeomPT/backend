from firebase_admin import initialize_app, credentials, firestore, storage
from flask import request, jsonify
import uuid
from io import BytesIO

# Firebase initialization
cred = credentials.Certificate(
    "hackmit2024-69f95-firebase-adminsdk-p2u0c-789554acbf.json"
)
initialize_app(cred, {"storageBucket": "hackmit2024-69f95.appspot.com"})

db = firestore.client()


# Generates a unique user ID
def generateUserId(name):
    return name + "_" + str(uuid.uuid4()).replace("-", "")[:8]


# Error handling wrapper
def errorWrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    wrapper.__name__ = func.__name__
    return wrapper


# Firebase app routes
def loadFirebaseFromApp(app):
    @app.route("/api/all-users", methods=["GET"])
    @errorWrapper
    def getUserIds():
        users = db.collection("users").get()
        data = [user.to_dict() for user in users]
        return jsonify(data), 200

    @app.route("/api/users", methods=["POST"])
    @errorWrapper
    def addUser():
        requestData = request.json
        userId = generateUserId(
            requestData["name"]
        )  # Fixed typo here (gerneate -> generate)
        name = requestData["name"]
        role = requestData["role"]
        relatedIds = requestData["relatedIds"]

        data = {
            "userId": userId,
            "name": name,
            "role": role,
            "relatedIds": relatedIds,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

        db.collection("users").document(userId).set(data)
        return jsonify({"message": "User added successfully"}), 200

    @app.route("/api/users/<userId>/<workout>", methods=["POST"])
    @errorWrapper
    def addGraphData(userId=None, workout=None):
        requestData = request.json
        value = requestData["value"]
        imageFile = requestData["imageFile"]  # Changed to file input for image
        videoUrl = requestData["videoUrl"]

        # Save the image to Firebase storage
        imageUrl = saveFile(userId, "images", f"{workout}_image.jpg", imageFile)

        # Add graph data to Firestore
        userRef = db.collection("users").document(userId)
        graphDataRef = userRef.collection(workout)

        data = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "value": value,
            "imageUrl": imageUrl,
            "videoUrl": videoUrl,  # Placeholder for video
        }

        graphDataRef.add(data)
        return (
            jsonify({"message": f"Graph data added for {userId} for {workout}"}),
            200,
        )


# Helper function to save file to Firebase Storage and return public URL
def saveFile(userId, fileType, fileName, fileBytes: BytesIO):
    bucket = storage.bucket()
    blob = bucket.blob(f"{fileType}/{userId}/{fileName}")
    blob.upload_from_file(fileBytes)
    blob.make_public()

    return blob.public_url
