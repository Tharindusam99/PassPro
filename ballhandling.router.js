const express = require("express");
const ballhandlingController = require("../controllers/ballhandling.controller");
const authenticateUser = require("../middleware/authMiddleware");

const router = express.Router();

// Upload ball handling videos
router.post("/upload", authenticateUser, ballhandlingController.uploadBallHandlingVideos);

// Get uploaded videos
router.get("/videos/:userId", ballhandlingController.getBallHandlingVideos);

// Get matching percentages
router.get("/matching/:userId", ballhandlingController.getBallHandlingMatchingPercentages);

// Analyze uploaded videos
router.post("/analyze/:id", authenticateUser, ballhandlingController.analyzeBallHandling);

// Delet records
router.delete("/delete/:id", authenticateUser, ballhandlingController.deleteBallHandling);

// Find top players in ball handling
router.post("/top-players", authenticateUser, ballhandlingController.getTopUsersByBallHandling);

// Get suggestions for attack analysis
router.get("/suggestions", authenticateUser,  ballhandlingController.ballHandlingSuggestions);


module.exports = router;
