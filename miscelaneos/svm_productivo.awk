# Funcion que dados dos vectores realiza su producto punto
function prod_punto(v1,v2,len)
{
  i=1;
  acum=0;
  while (i<=len){
  	acum+=v1[i]*v2[i];
  	i++;
  }
  return acum;   
}

# Cacula W^T.Z
function suma_sv(a,n,t,s,l){
	sc=1;
	ac=0;
	while(sc<=n){
		rpc=prod_punto(t,s[sc],l);
		ac+=a[sc]*rpc;
		sc++;
	}
	return ac;
}

BEGIN{
	tp=0;
	tn=0;
	fp=0;
	fn=0;

	sv_flag=0;
	sv_cont=0;
	
	# Carga en la matriz 'sv' y en el arreglo 'alfa' el modelo obtenido con svm_train
	while(( getline line<"train.scale.model") > 0 ) {
		# Separa la linea en registros via 'vars[i]'
		split(line,vars);

		#Si ya procese todo el header entra a cargar los vectores de soporte
     		if(sv_flag==1){
			#ACA SE PROCESAN LOS SVs
			sv_cont++;
			len=split(line,svt);
			# Hacer un while que reconstruye la aridad original (95 en este caso) porque vienen en formato sparse
			i=1;
			while(i<=95){
				sv[sv_cont][i]=0.0;
				i++;
			}

			# Cargo el valor de alfa del vector de soporte que se esta procesando
			alfa[sv_cont]=svt[1];

			# Cargo en la matriz sv los valores de features para el vector de soporte actual
			i=2;
			while(i<=len){
				split(svt[i],tmp,":");
				sv[sv_cont][tmp[1]]=tmp[2];
				i++;
			}
		}
		else{	
				#Aun hay header por procesar
				if(line=="SV"){
					sv_flag=1;
				}
				else{
						if(vars[1]=="rho"){
							b=-1*vars[2];
						}; 
						if(vars[1]=="nr_sv"){
							sv_pos=vars[2]; 
							sv_neg=vars[3];
						};
					}
			}
  	}
}
{
	lon=split($0,test_inst);

	# Dejo en el arreglo 'test' la instancia que se quiere clasificar
	c=2;
	while(c<=lon){
		split(test_inst[c],tmp_t,":");
		test[tmp_t[1]]=tmp_t[2];
		c++;
	}
	
	output=0;

	# Invoco a 'suma_sv' 
	ns = sv_pos + sv_neg;
	output=suma_sv(alfa,ns,test,sv,95);

	# Le sumo el valor del bias
	output+=b;

	# Clasifico y calculo estadisticas. 
	# Nota si no tuviera la etiqueta de clase no podria calcular la estadisticas y en la linea 83 'c' deberia inicializar se 1.
	if(output>0) {
		print "1";
		if(test_inst[1]=="1") tp++;
		else fp++;
	}
	else {
		print "-1";
		if(test_inst[1]=="1") fn++;
		else tn++;
	}

}
END{
	print "";
	print "----------------------";
	print "Performance Statistics";
	print "----------------------";
	printf "TP=%s\n",tp;
	printf "TN=%s\n",tn;
	printf "FP=%s\n",fp;
	printf "FN=%s\n",fn;
}
