#!/usr/bin/bash
#

echo Running wgetdata script
sleep 1

sc='EUVI'
outroot_img="/Volumes/DataDisk/$sc"
ycnt=0
mcnt=0
dcnt=0
wcnt=0

years=(2016 2017 2018 done)

months=(01 02 03 04 05 06 07 08 09 10 11 12 done)

wave=(171_A 195_A 284_A 304_A done)


cd $outroot_img
sleep 1
pwd
sleep 2

if [ ${?} ] 
then 
# we are in correct directory

while [ ${years[$ycnt]} != "done" ]
do
    yr=${years[$ycnt]}

    while [ ${months[$mcnt]} != "done" ]
    do
	    mon=${months[$mcnt]}
	    	
        if  [ $mon == 01 ] ||  [ $mon == 03 ] ||  [ $mon == 05 ] ||  [ $mon == 07 ] ||  [ $mon == 08 ] ||  [ $mon == 10 ] ||  [ $mon == 12 ]
        then
            days=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 done)
        elif  [ $mon == 04 ] ||  [ $mon == 06 ] ||  [ $mon == 09 ] ||  [ $mon == 11 ]
        then
            days=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 done)
        else
            days=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 done)
        fi

        while [ ${days[$dcnt]} != "done" ]
        do
            day=${days[$dcnt]}
	        
	        while [ ${wave[$wcnt]} != "done" ]
             do
	            wv=${wave[$wcnt]}
	            
                cd $outroot_img/$yr/$wv
	            
	            pwd
	            
	            echo Retrieving $sc $yr $mon $day $wv
                
                wget -m -r -nH --no-parent --cut-dirs=7 -nv -A '*.fts.gz' http://sd-www.jhuapl.edu/secchi/wavelets/fits/$yr$mon/$day/$wv/
                
	            wcnt=`expr $wcnt + 1`

	        done
	        wcnt=0
	        dcnt=`expr $dcnt + 1`
	    done
	    wcnt=0
	    dcnt=0
	    mcnt=`expr $mcnt + 1`
	done
	wcnt=0
	dcnt=0
	mcnt=0
	ycnt=`expr $ycnt + 1`
	echo Waiting 10 sec before getting ${yr[$ycnt]}
	sleep 10
done
unset wave[4]
echo done with $sc ${wave[*]}
fi
exit
