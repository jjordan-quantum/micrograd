from graphviz import Digraph

dot = Digraph(comment='RZX Project Flow')

dot.node('A', 'Start')
dot.node('B', 'Mobilize Equipment')
dot.node('C', 'Perform Excavation')
dot.node('D', 'Install Conduit')
dot.node('E', 'Backfill & Test')
dot.node('F', 'Demobilize')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

# Render and open automatically
dot.render('rzx_project_flow.gv', view=True)