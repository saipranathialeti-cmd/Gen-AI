"""Tiny training text shared by all labs."""

ENGLISH = [
    "Artificial intelligence is changing the world.",
    "Students are learning computer science.",
    "The weather is good today.",
    "I am going to college.",
    "Machine learning needs a lot of data.",
    "The model reads text as tokens.",
    "Tokenization splits text into small pieces.",
    "Language models learn from large text.",
]

TELUGU = [
    "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
    "విద్యార్థు లు కంప్యూటర్ నేర్చుకుంటున్నారు.",
    "ఈ రోజు వాతావరణం బాగుంది.",
    "నేను కళాశాలకు వెళ్తు న్నాను.",
]

HINDI = [
    "कृ त्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
    "छात्र कं प्यूटर सीख रहे हैं।",
    "आज मौसम अच्छा है।",
    "मैं कॉलेज जा रहा हूँ।",
]

TEST_TEXTS = {
    "English": "Artificial intelligence is changing the world.",
    "Telugu": "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
    "Hindi": "कृ त्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
}

# Balance the training data across languages.
TRAINING_LINES = ENGLISH + TELUGU * 2 + HINDI * 2