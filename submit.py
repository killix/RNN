import sys
import numpy as np

answer_letters = list('abcde')

if __name__ == '__main__':
    f = open(sys.argv[1])
    question_answers = []
    for answer_options in zip(*[iter(f)] * 5):
        log_prods = []
        for ao in answer_options:
            logp = np.sum(np.log(np.array(
                [float(p) for p in ao.rstrip().split(',')]
            )))
            log_prods.append(logp)
        answer = answer_letters[np.argmax(log_prods)]
        question_answers.append(answer)

    with open('submission_final.csv', 'w') as f:
        print('Id,Answer', file=f)
        for qid, qans in enumerate(question_answers, 1):
            print('%d,%s' % (qid, qans), file=f)
