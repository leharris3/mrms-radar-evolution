# TODO
  
1. Gather a small dataset of radar data/{watch, warning, report} data to generate (input, target) pairs.
2. Create a model that predicts P(tornado_warn <= 25 miles | loc, radar)
    - Will the confidence intervals produced by this model be meaningful/interpretable?
    - Easy solution:
        - Reformulate so that model predicts tornado warn <= (50, 25, 15, 5, 1) miles
        - Interpretability is no longer a big issue; model can opperate end-to-end
