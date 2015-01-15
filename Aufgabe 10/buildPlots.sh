#!/bin/bash

for file in ./*.dat
do
	echo "Plot "$file
	values=$(awk '{match($0,"[0-9]+ [0-9]+ [0-9]+",a)}END{print a[0]}' $file)
	proc=$(echo ${values} | awk '{print $1}')
	intr=$(echo ${values} | awk '{print $3}')
	type="Jacobi"
	funcNum=2
	typeNum=2
	iter=1500
	term=2
	col=2
	colName="Knoten"
	test="Communikations Test"
	if [[ $file =~ "GS.dat" ]]
	then
		type="Gauss Seidel"
		typeNum=1
	fi
	if [[ $file =~ "COMMUNICATION" ]]
	then
		col=2
		colName="Knoten"
		funcNum=1
	fi
	if [[ $file =~ "STRONG" ]]
	then
		col=1
		colName="Prozesse"
		iter="1000"
		test="Strong Scaling Test"
	fi
	if [[ $file =~ "WEAK" ]]
	then
		col=1
		colName="Prozesse"
		iter="1500"
		test="Weak Scaling Test"
	fi
	if [[ $file =~ "_A_" ]]
	then
		term=1
		iter="3.3504e-05"
	fi

	gnuplot <<- EOF
        	set xlabel "${colName}"
        	set ylabel "Zeit in Sekunden"
		set title "${test} ${type} mit ${intr} Interlines"
       	 	set term png
        	set output "${file}.png"
        	plot "${file}" using $col:4 with linespoints title "mpiexec -n #Processcount ./partdiff-par 1 ${typeNum} ${intr} ${funcNum} ${term} ${iter}"
		save "${file}.gnuplot"
		EOF
done
