from numpy import log, exp

Let_spam = 19
Let_no_spam = 17
words_spam = 71
words_no_spam = 125
P_let_spam = Let_spam / (Let_no_spam + Let_spam)
print('Вероятность письма оказаться спамом:', round(P_let_spam,3))

# Purchase Bill Gift Unlimited Bonus Prize Cash
r = 2
V = 10
words = [[9,0],[4,16],[4,1],[5,13],[1,5]]
F_let_spam = 0
F_let_no_spam = 0
for i in range(0,5):
    F_let_spam += log((1 + words[i][0]) / (r + V + words_spam))
    F_let_no_spam += log((1 + words[i][1]) / (r + V + words_no_spam))
F_let_spam += 2 * log((1 / (r + V + words_spam))) + log(P_let_spam)
F_let_no_spam += 2 * log((1 / (r + V + words_no_spam))) + log(1 - P_let_spam)
P_spam = 1 / (1 + exp(F_let_no_spam - F_let_spam))
print('Вероятность классифицировать письмо как спам:', round(P_spam,3))
print('F(спам):', round(F_let_spam,3), ';', 'F(не спам):', round(F_let_no_spam,3))


Pps = (1 + 9) / (r + V + 71)
Pbs = (1 + 4) / (r + V + 71)
Pgs = (1 + 4) / (r + V + 71)
Pbos = (1 + 5) / (r + V + 71)
Pprs = (1 + 1) / (r + V + 71)
Fpsp = log(P_let_spam) + log(Pps) + log(Pbs) + log(Pgs) + log(Pbos) + log(Pprs) + 2 * log(1 / (r + V + 71))
Ppns = (1 + 0) / (r + V + 125)
Pbns = (1 + 16) / (r + V + 125)
Pgns = (1 + 1) / (r + V + 125)
Pbons = (1 + 13) / (r + V + 125)
Pprns = (1 + 5) / (r + V + 125)
Fpnsp = log(1 - P_let_spam) + log(Ppns) + log(Pbns) + log(Pgns) + log(Pbons) + log(Pprns) + 2 * log(1 / (r + V + 125))
P_spam1 = 1 / (1 + exp(Fpnsp - Fpsp))
print(round(Fpsp,3), round(Fpnsp,3),round(P_spam1,3))