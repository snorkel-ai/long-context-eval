### Data and Task format requirements

To run on your own data, add documents to a `data` folder. (As a default, we use `Unstructured` to extract text from documents. If you have PDF/HTML/etc. documents, install the necessary `Unstructured` libraries.)

The QA task needs to be in the following format, and contain the keys `question`, `answer` and `answer_doc`.

```zsh
{
    "0": {
        "question": "What are the nine standard H2H categories to analyze in Fantasy Basketball?",
        "answer": "Points, rebounds, assists, steals, blocks, three-pointers made, field goal percentage, free throw percentage, and turnovers.",
        "answer_doc": "How to Nail Your Fantasy Basketball Draft and Identify Top Center Position Sleepers.txt"
    },
    "1": {
        "question": "What alternative solution did Justin Pritchett find for Vincent Gonzales in the trespassing incident?",
        "answer": "Pritchett purchased a new gym membership for Gonzales.",
        "answer_doc": "How to Respond to Trespassers as a Public Safety Official While Upholding Community Engagement.txt"
    },
    "2": {
        "question": "Where did Giannis Antetokounmpo grow up and develop his passion for basketball?",
        "answer": "Giannis Antetokounmpo grew up in Greece.",
        "answer_doc": "How to Become a Professional Basketball Player Like Giannis Antetokounmpo.txt"
    },
}