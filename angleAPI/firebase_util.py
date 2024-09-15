from firebase_admin import initialize_app, credentials, firestore, storage
from flask import request, jsonify
import uuid
from io import BytesIO

cred = credentials.Certificate(
    "hackmit2024-69f95-firebase-adminsdk-p2u0c-789554acbf.json"
)
initialize_app(cred, {"storageBucket": "hackmit2024-69f95.appspot.com"})

db = firestore.client()


def gerneateUserId(name):
    return name + "_" + str(uuid.uuid4()).replace("-", "")[:8]


def errorWrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    wrapper.__name__ = func.__name__
    return wrapper


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
        userId = gerneateUserId(requestData["name"])
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

    @app.route("/api/users/<userId>", methods=["GET"])
    @errorWrapper
    def getUser(userId=None):
        userRef = db.collection("users").document(userId)
        userDoc = userRef.get()

        print("sending back user")

        if userDoc.exists:
            return jsonify(userDoc.to_dict()), 200
        else:
            return jsonify({"error": f"No such user with ID: {userId}"}), 404

    @app.route("/api/users/<userId>/<workout>", methods=["POST"])
    @errorWrapper
    def addGraphData(userId=None, workout=None):
        requestData = request.json
        value = requestData["value"]
        imageUrl = requestData["imageUrl"]
        videoUrl = requestData["videoUrl"]

        userRef = db.collection("users").document(userId)
        graphDataRef = userRef.collection(workout)

        data = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "value": value,
            "imageUrl": imageUrl,
            "videoUrl": videoUrl,
        }

        graphDataRef.add(data)
        return (
            jsonify({"message": f"Graph data added for {userId} for {workout}"}),
            200,
        )

    @app.route("/api/users/<userId>/<workout>", methods=["GET"])
    @errorWrapper
    def getGraphDataEndpoint(userId=None, workout=None):
        userRef = db.collection("users").document(userId)
        graphData = userRef.collection(workout).get()

        if graphData:
            data = [data_point.to_dict() for data_point in graphData]
            return jsonify(data), 200
        else:
            return jsonify({"error": f"No graph data for user {userId}"}), 404

    # @app.route("/api/users/<userId>/images", methods=["POST"])
    # @errorWrapper
    # def addImage(userId=None):
    #     requestData = request.json
    #     imageData = requestData["imageData"]

    #     userRef = db.collection("users").document(userId)
    #     imagesRef = userRef.collection("images")

    #     data = {
    #         "imageData": imageData,
    #         "timestamp": firestore.SERVER_TIMESTAMP,
    #     }

    #     imagesRef.add(data)
    #     return (
    #         jsonify({"message": f"Image for {userId} added"}),
    #         200,
    #     )

    # @app.route("/api/users/<userId>/images", methods=["GET"])
    # @errorWrapper
    # def getImages(userId=None):
    #     userRef = db.collection("users").document(userId)
    #     images = userRef.collection("images").get()

    #     if images:
    #         data = [image.to_dict() for image in images]
    #         return jsonify(data), 200
    #     else:
    #         return jsonify({"error": f"No images for user {userId}"}), 404


def addGraphData(userId, value, workoutName, imageUrl, videoUrl):
    userRef = db.collection("users").document(userId)
    graphDataRef = userRef.collection(workoutName)

    data = {
        "timestamp": firestore.SERVER_TIMESTAMP,
        "value": value,
        "imageUrl": imageUrl,
        "videoUrl": videoUrl,
    }

    graphDataRef.add(data)


def saveFile(userId, fileType, fileName, fileBytes: BytesIO):
    bucket = storage.bucket()
    blob = bucket.blob(f"{fileType}/{userId}/{fileName}")
    blob.upload_from_file(fileBytes)
    blob.make_public()

    return blob.public_url
