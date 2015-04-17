#data path
#added a comment
data_path <- 'C:/Users/DKenefick/Desktop/Student/astudentData.csv'
out_path <- '/Desktop/Python/Students/model_results.csv'

main <- function(d_path, o_path){
	# pull in the data
	data = read.csv(d_path,header=TRUE)

	# reshapre and clean
	data_wide <-reshape(data,idvar = 'user_id',timevar='question_id',direction='wide')
	rownames(data_wide) <- data_wide$'user_id'
	data_wide$user_id <- NULL

	# fit a two parameter IRT model on the responses, ignoring missing
	# data (default). 
	result=ltm(data_wide~z1)

	#fit a three parameter model (including a guessing paramter)
	#assuming the guessing paramter is flexible
	#ignoring missing data
	result2 = tpm(data_wide,type='latent.trait')

	#anova of two models
	#suggests two parameter model is better
	anova(result,result2)

	#clean up data and export
	out = coef(result,prob= TRUE, order = TRUE)
	out = data.frame( as.numeric(sub('correct.','',rownames(out))) , out)
	rownames(out)<- NULL
	names(out)[1] <- 'question_id'
	names(out)[4] <- 'p_guess'

	write.table(out,file=o_path ,sep = ',',quote=FALSE,row.names=FALSE)
}


#require ltm package
if(require("ltm")){
    print("ltm is loaded correctly")
    main(d_path=data_path, o_path = out_path)	
} else {
    print("trying to install ltm")
    install.packages("ltm")
    if(require(ltm)){
        print("ltm installed and loaded")
        main(d_path=data_path, o_path = out_path)
    } else {
        stop("could not install ltm")
    }
}


