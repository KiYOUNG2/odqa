from solution.odqa import ODQA


def make_answer(question, context):
    odqa = ODQA()

    return odqa.answer(query=question, context=context)

if __name__ == "__main__":
    question = input()
    context = input()
    answer, answerable = make_answer(question, context)
    print(answer)