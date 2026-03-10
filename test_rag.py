from rag_pipeline import ask_question

while True:
    question = input("Ask something: ")

    if question == "exit":
        break

    answer = ask_question(question)

    print("\nAnswer:", answer)