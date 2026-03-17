from khundech.auto_learning import parse_auto_learning_add_intent

text = '''add more autolern
purpose : lern coding for improve my self
init question : what is python and how can i use for deeply for ai service project and give me example of code
interval : 10minute
'''

print(parse_auto_learning_add_intent(text))
