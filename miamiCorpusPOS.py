from pysentimiento import create_analyzer

pos_analyzer = create_analyzer("pos", lang="es")

print(pos_analyzer.predict("Quiero que esto funcione correctamente!"))
 
 