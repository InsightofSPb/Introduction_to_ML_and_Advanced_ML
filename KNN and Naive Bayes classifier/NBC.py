from numpy import *

N = 70  # кол-во писем
P = 6  # "приз" в спаме
Pc = 5  # "приз" не в спаме
C = 30  # кол-во спамных писем
print('Количество спамных писем:', P / C)
# вероятность того, что письмо есть спам, если содержит "приз"
# P(A|B) = (P(A) * P(B|A)) / P(B)
SP = (C / N) * (P / C) / ((P + Pc) / N)
print(SP)

Let = [11,26]
Letsp = Let[0]/ sum(Let)
letspln = log(Letsp)
print(round(Letsp,3),round(letspln,3))

r = 1  # dollar, number of words not in dictionary
V = 8  # size of the dictionary
Spam = 69  # total number of spam words (words in spam letters)
Clear = 48  # total number of clear words
wins = 6  # win in spam
winc = 3  # win in clear
millions = 10
millionsc = 5
P_win_spam = round((r + wins) / (V + r + Spam),4)
P_dollars_spam = round((r) / (r + V + Spam), 4)
print(P_win_spam)
print(P_dollars_spam)
P_million_spam = round((r + 10) / (r + V + Spam),3)
F_spam = log(Let[0] / sum(Let)) + log((r + wins) / (r + V + Spam)) + log((r + millions) / (r + V + Spam)) + log((r) / (r + V + Spam))
F_clear = log(Let[1] / sum(Let)) + log((r + winc) / (r + V + Clear)) + log((r + millionsc) / (r + V + Clear)) + log((r) / (r + V + Clear))
P_spam_letter = 1 / (1 + exp(F_clear - F_spam))
print(round(F_spam,3), round(F_clear,3), round(P_spam_letter,3))

