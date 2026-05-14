from uniplot import plot

def cummulative_rank(ranks):

    valid_ranks = [r for r in ranks if r != -1]
    
    if not valid_ranks:
        print("No hay rankings dentro del tok. O hiciste un bug o --eval-size es muy pequeño.")
        return
    
    max_rank = max(valid_ranks) + 1
    cumulative_acc = []
    for k in range(1, max_rank + 1):
        acc_at_k = sum(1 for r in valid_ranks if r < k) / len(ranks)
        cumulative_acc.append(acc_at_k)
    
    plot(cumulative_acc, title='P(Ranking <= X)', lines=True)
