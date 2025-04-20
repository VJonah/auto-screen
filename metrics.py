# std lib imports
from collections import Counter
from typing import Callable
from math import sqrt

# pkg imports
import dspy
from tqdm import tqdm

def precision(c: Counter) -> float:
    pos_total = c['TP'] + c['FP']
    if pos_total:
        return c['TP']/pos_total
    return float('nan')
    
def recall(c: Counter) -> float:
    divisor = c['TP'] + c['FN']
    if divisor:
        return c['TP']/divisor
    return float('nan')
    
def f1score(c: Counter) -> float:
    prec, rec = precision(c), recall(c)
    if prec and rec:
        return 2*(prec*rec)/(prec+rec)
    return float('nan')
    
def specificity(c: Counter) -> float:
    n = sum(c.values())
    if n:
        return 1-(c['FP']/n)
    return float('nan')

def mcc(c: Counter) -> float:
    tp, tn, fp, fn = c['TP'], c['TN'], c['FP'], c['FN']
    dividend = tp*tn - fp*fn
    divisor = sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if divisor:
        return dividend/divisor
    return float('nan')
    
def validate_all_criteria(example,
                          pred,
                          trace=None) -> str | bool:
    """
    Validates a collection of inclusion/exclusion 
    criteria and their satisfiability, only if all 
    criteria are True/satisfied.
    """
    pred.relevant = all(pred.satisfied.values())
    return confusion_validate(example, pred, trace=trace)
    
def confusion_validate(example, 
                       pred, 
                       trace=None) -> str | bool:
    """
    Validates a classification prediction using
    confusion matrix cells, i.e.:
    
    - True Positive
    - True Negative
    - False Positive
    - False \negative
    """
    if trace is None:
        if example.relevant and pred.relevant:
            return "TP" # return True Positive
        elif not example.relevant and not pred.relevant:
            return "TN" # return True Negative
        elif not example.relevant and pred.relevant:
            return "FP" # return False Positive
        else:
            return "FN" # return False Negative
    else:
        return example.relevant == pred.relevant

def batch_sr_eval(program: dspy.Program,
                  devset: dict[str,list[dspy.Example]],
                  eval_func: Callable = f1_evaluate,
                  metric: Callable = validate_all_criteria): -> None:
    """
    Evaluates a batch of Systematic Reviews one at at time.

    Keyword arguments:
    program   --  the dspy program/module to evaluate
    devset    --  a dict of id-devset pairs to evaluate against
    eval_func --  the specific evaluation function to use 
                  (default f1_evaluate)
    metric    --  the specific metric to evaluate predictions
                  against (default validate_all_criteria)
    """
    
    for batch_id, data in devset.items():
        sr_title, data = data[0], data[1:][0]
        print(f"Batch: {batch_id}")
        eval_fun(program(sr_title), data, metric)
    
def f1_evaluate(program: dspy.Program,
               devset: list[dspy.Example],
               metric: Callable = confusion_validate) -> None:
    """
    Evaluate a program with confusion matrix metrics.

    Keyword arguments:
    program -- the dspy program to evaluate
    devset  -- the development dataset to evaluate against
    metric  -- the specific metric to evaluate predictions 
               against (default confusion_validation)
    """
    c = Counter()
    with tqdm(total=len(devset),
              bar_format="{postfix[0]} {postfix[1][value]:.3f} {l_bar}{bar}'| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]'",
              postfix=["F1 Score:", {'value': float('nan')}]) as t:
        for x in devset:
            pred = program(**x.inputs())
            score = metric(x, pred)
            c[score] += 1
        
            # scores
            prec = precision(c)
            rec = recall(c)
            f1 = f1score(c)
            
            # update progress bar
            t.postfix[1]['value'] = f1
            t.update()

    print(f"Confusion Matrix: {c}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"MCC: {mcc(c):.3f}")
    print(f"Specificity: {specificity(c):.3f}")