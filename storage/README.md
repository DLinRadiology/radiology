Status dabar:

main.py callina heartseg.py, kuris issegmentuoja sirdi, ir savina pries segmentacija + segmentuota image i ta pati folderi.
main.py callina cool.py, kuris pasako ar image frontal ar lateral.


Pirmas goal:

Kad website/flask/html interface (nzn kaip pavadint, toliau vadinsiu tiesiog "page") - rodytu rezultatinius images (kurie kolkas kaip minejau savinasi tiesiog folderyje).

Tam pasiekti as I see it yra 2 budai:
	
	upload images to cloud (tfbucket), getting ju link ir embeddint.
	shortcut budas - tiesiogiai embeddint is matplotlibo, pvz. mix of:
		https://gist.github.com/wilsaj/862153
		https://stackoverflow.com/questions/20107414/passing-a-matplotlib-figure-to-html-flask


Galutinis goal (kolkas):

Pirmas page su 2 mygtukais: "classify chest x-ray to frontal or lateral" ir "segment heart on frontal chest x-ray"

Pirmas mygtukas ("classify...") atidaro dabartini page, kur tiesiog galima uploadint image ir returnina atsakyma.

Antras mygtukas ("segment....") atidaro page kur yra panasus paprastas page uploadinimui. Uploadinus returninami abu images (pries ir po segmentacijos)...padidinti, kad matytusi normaliau nei 96x96 pixels. + Parasyta "Please rate this segmentation:" ir 3 mygtukai: "Good", "Somewhat OK" ir "Bad". Nesvarbu kuri paspaudzia, atsiranda text "Thank you for your help! Back to first page to try again: [linkas i pirma page]. Back to [link: dlinradiology.wordpress.com]." Ir tada - musu cloude tfbuckete butu 3 folders pavadinti "heartseg_good", "heartseg_ok", "heartseg_bad". Priklausomai nuo paspausto mygtuko - uploadintas image butu uploadintas i atitinkama folderi, kartu su segmentuotu image.

Butu nice jei pervadintum "storage" i kazka normalesnio, taip pat "cool.py" galetu but pvz. classify_front_lat.py or something ir tt... as bijau daryt per daug changes nes nesu tikras ar iseitu restorinti man :D.


Comments:
Kad uploadint i serveri tai main.py turi daugmas ko reikia. Jei nebutu to "premature" return teksto, tai butu uploadinta i bucketa. As return padares auksciau, nes pas mane locally runninant su serveriu/bucket nesusijungia (tam reikia additional steps) ir stoppinasi, bet anksciau kiek bandes tai veikia.
Manau butu geriausiai jei tu butum in charge to puslapio ir moketum uploadint viska i clouda. Nes jei tu sugebetum padaryti tai kas kolkas yra goal, tai veliau prideti nauja mygtuka kai butu naujas tool manau turetu buti tikrai lengva. Ir tada as nepridaryciau tokios makalynes...

Veliau reikes pagalvoti kaip optimizuoti uploadinima...nes kai prades uploadint CT ar MRI failus, tai ten yra multiple files, dydis variable, bet gal total 50 mb, tai nei daug nei mazai...tik va nzn kaip tai, kad multiple files reikes handlinti. Galima jei ka paprasyt, kad pirma zippintu...bet cia jau veliau. 

ACIU! Kai darysim startup uz kokiu metu, nemanyk, kad busi left out aisku :)