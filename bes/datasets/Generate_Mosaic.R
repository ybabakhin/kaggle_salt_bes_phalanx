require(data.table)
path='/Volumes/Media/DataScience/Kaggle/Salt/all/'
output_path = '/Volumes/Media/DataScience/Kaggle/Salt/kaggle-salt/data/mos_numpy_v2/'

depth=fread(paste0(path,'depths.csv'))
rle=fread(paste0(path,'train.csv'))
stest=fread(paste0(path,'sample_submission.csv'))
lmod=paste0(rle$id,'.png')
ltest=paste0(stest$id,'.png')


read_img<-function(fname,spath){
  im=load.image(paste0(path,spath,'/images/',fname))
  return(as.vector(im[,,1,1]))
}
read_mask<-function(fname,spath){
  im=load.image(paste0(path,spath,'/masks/',fname))
  return(as.vector(im[,,1,1]))
}

library(imager)
library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)

system.time({
  train <- foreach(i = 1:length(lmod), .packages = c('imager'),.combine='rbind') %dopar% {
    read_img(lmod[i],'train')
  }
})

system.time({
  masks <- foreach(i = 1:length(lmod), .packages = c('imager'),.combine='rbind') %dopar% {
    read_mask(lmod[i],'train')
  }
})

system.time({
  test <- foreach(i = 1:length(ltest), .packages = c('imager'),.combine='rbind') %dopar% {
    read_img(ltest[i],'test')
  }
})
stopCluster(cl)


di=101
li=nrow(train)+nrow(test)
all_arr=array(rbind(train,test),dim=c(li,di,di))

all_u_ex=scale(2*all_arr[,1,]-all_arr[,2,])
all_d_ex=scale(2*all_arr[,101,]-all_arr[,100,])

all_u_ex=t(apply(all_u_ex,1,function(x)scale(x,center = T,scale = T)))
all_d_ex=t(apply(all_d_ex,1,function(x)scale(x,center = T,scale = T)))

all_u_ex[is.na(all_u_ex)]=0
all_d_ex[is.na(all_d_ex)]=0

all_l_ex=scale(2*all_arr[,,1]-all_arr[,,2])
all_r_ex=scale(2*all_arr[,,101]-all_arr[,,100])

all_l_ex=t(apply(all_l_ex,1,function(x)scale(x,center = T,scale = T)))
all_r_ex=t(apply(all_r_ex,1,function(x)scale(x,center = T,scale = T)))

all_l_ex[is.na(all_l_ex)]=0
all_r_ex[is.na(all_r_ex)]=0


require(FNN)
system.time(neldu<-get.knnx(all_d_ex,all_u_ex,k=2))
system.time(nelud<-get.knnx(all_u_ex,all_d_ex,k=2))

system.time(nelr<-get.knnx(all_l_ex,all_r_ex,k=2))
system.time(nerl<-get.knnx(all_r_ex,all_l_ex,k=2))



gen_mosaic<-function(ad,ac){
  #generate candidates left-right
  
  dlr=cbind(as.data.table(nelr$nn.index),as.data.table(nelr$nn.dist))
  colnames(dlr)=c('i1','i2','d1','d2')
  dlr[,i0:=1:nrow(dlr)]
  dlr[,c:=1-d1/d2] # Compatibility
  
  
  drl=cbind(as.data.table(nerl$nn.index),as.data.table(nerl$nn.dist))
  colnames(drl)=c('i1','i2','d1','d2')
  drl[,i0:=1:nrow(drl)]
  drl[,c:=1-d1/d2]
  
  
  bb2=merge(dlr,drl,by.x='i0',by.y='i1')
  
  #filter by disimilarity and compatibility
  bb2=bb2[i0!=i1 & d1.x<ad & c.x>ac & c.y>ac]
  
  
  
  #find left-right strips 
  
  nt=nrow(all_arr)
  lcols=list()
  eval=array(F,nt)
  
  for(i in 1:nt){
    if(!eval[i]){
      lt=c(i)
      eval[i]=F
      cond=T
      i0=i
      while(cond){
        i1=bb2$i1[bb2$i0==i0]
        if(length(i1)==1 & length(which(lt==i1))==0){
          lt=c(i1,lt)
          i0=i1
          eval[i1]=T
        }
        else
          cond=F
      }
      cond=T
      i0=i
      while(cond){
        i1=bb2$i0[bb2$i1==i0]
        if(length(i1)==1  & length(which(lt==i1))==0){
          lt=c(lt,i1)
          i0=i1
          eval[i1]=T
        }
        else
          cond=F
      }
      if(length(lt)>1){
        #print(lt)
        lcols=append(lcols,list(lt))
      }     
    }
  }
  
  #Same for up-down
  
  ddu=cbind(as.data.table(neldu$nn.index),as.data.table(neldu$nn.dist))
  colnames(ddu)=c('i1','i2','d1','d2')
  ddu[,i0:=1:nrow(ddu)]
  ddu[,c:=1-d1/d2]
  
  
  
  dud=cbind(as.data.table(nelud$nn.index),as.data.table(nelud$nn.dist))
  colnames(dud)=c('i1','i2','d1','d2')
  dud[,i0:=1:nrow(dud)]
  dud[,c:=1-d1/d2]
  
  bb=merge(ddu,dud,by.x='i0',by.y='i1')
  
  bb=bb[i0!=i1 & d1.x<ad & c.x>ac & c.y>ac]
  
  
  #Generate up-down stripes  
  nt=nrow(all_arr) 
  lrows=list()
  eval=array(F,nt)
  for(i in 1:nt){
    if(!eval[i]){
      lt=c(i)
      eval[i]=F
      cond=T
      i0=i
      while(cond){
        i1=bb$i1[bb$i0==i0]
        if(length(i1)==1 & length(which(lt==i1))==0){
          lt=c(i1,lt)
          i0=i1
          eval[i1]=T
        }
        else
          cond=F
      }
      cond=T
      i0=i
      while(cond){
        i1=bb$i0[bb$i1==i0]
        if(length(i1)==1  & length(which(lt==i1))==0){
          lt=c(lt,i1)
          i0=i1
          eval[i1]=T
        }
        else
          cond=F
      }
      if(length(lt)>1){
        #print(lt)
        lrows=append(lrows,list(lt))
      }     
    }
  }
  
  #Finally combine rows and colums
  
  rc=array(0,dim=c(nt,2))
  
  for(i in 1:length(lrows))rc[lrows[[i]],1]=i
  for(i in 1:length(lcols))rc[lcols[[i]],2]=i
  
  rc=as.data.table(rc)
  
  bt=rbind(bb,bb2)
  
  require(igraph)
  gra3=graph_from_edgelist(as.matrix(bt[,.(i0,i1)]))
  clu=components(gra3)
  
  ls=lapply(1:clu$no,function(x)which(clu$membership==x))
  
  lls=unlist(lapply(ls,length))
  
  dd=data.table(i=1:length(lls),l=lls)
  
  dd=dd[order(-l)]
  list(dd,ls,rc,lrows,lcols)
}

# **********************************
disim=10
compat=.25
ldd=gen_mosaic(disim,compat)

dd=ldd[[1]] # All mosaics ordererd by number of images
ls=ldd[[2]]
rc=ldd[[3]]
lrows=ldd[[4]] 
lcols=ldd[[5]]

dd

# Draw
require(imager)


# From seed, complete row and column
complete<-function(se,mat){
  
  ir0=unlist(rc[se,1])
  ic0=unlist(rc[se,2])
  
  x0=mat[which(mat[,1]==se),2]
  y0=mat[which(mat[,1]==se),3]
  
  if(ir0>0){
    r0=lrows[[ir0]]
    
    for(i in 1:length(r0)){
      if(length(which(mat[,1]==r0[i]))==0)mat=rbind(mat,c(r0[i],x0-which(r0==se)+i,y0,0))
    }
  }
  if(ic0>0){
    c0=rev(lcols[[ic0]])
    for(i in 1:length(c0)){
      if(length(which(mat[,1]==c0[i]))==0)mat=rbind(mat,c(c0[i],x0,y0-which(c0==se)+i,0))
    }
  }
  mat[which(mat[,1]==se),4]=1
  unique(mat)
}


# Complete iteratively a mosaic from a seed image
gen_mos<-function(se){
  mat=array(c(se,0,0,0),dim=c(1,4))
  
  while(sum(mat[,4])<nrow(mat)){
    for(i in 1:nrow(mat)){
      if(mat[i,4]==0)mat=complete(mat[i,1],mat)
    }
  }
  
  mat[,2]=mat[,2]-min(mat[,2])
  mat[,3]=mat[,3]-min(mat[,3])
  mat
}

plmos<-function(mat){
  
  lx=max(mat[,2])+1
  ly=max(mat[,3])+1
  
  n=nrow(mat)
  
  mos=array(0,dim=c(lx*101,ly*101))
  mosmas=mos*0
  for(k in 1:n){
    fi=mat[k,2]
    co=mat[k,3]
    id1=mat[k,1]
    mos[fi*101+c(1:101),co*101+c(1:101)]=all_arr[id1,,]
    mosmas[fi*101+c(1:101),co*101+c(1:101)]=all_arr[id1,,]
    if(id1<=nrow(train))mosmas[fi*101+c(1:101),co*101+c(1:101)]=masks[id1,]
  }
  
  mos=add.color(as.cimg(mos))
  R(mos)=as.cimg(mosmas)
  plot(as.cimg(mos),)
}

#for(ii in 1:10){
#  mat=gen_mos(ls[[dd[ii,i]]][1])
#  plmos(mat)
#}

trid <- data.frame(rle$id, stringsAsFactors=FALSE)
colnames(trid) <- c("image")
teid <- data.frame(stest$id, stringsAsFactors=FALSE)
colnames(teid) <- c("image")
all_ids=rbind(trid, teid)

domap<-function(mat){
  mat2 = mat
  lx=max(mat[,2])+1
  ly=max(mat[,3])+1
  n=nrow(mat)
  for(k in 1:n){
    id1=mat[k,1]
    mat2[k,1] = all_ids[id1,1]
    if(id1<=4000) {
      mat2[k,4] = 'train'
    }
    else {
      mat2[k,4] = 'test'
    }
  }
  df = data.frame(mat2)
  colnames(df) <- c("image", "lx", "ly", "dataset")
  df$lx = as.numeric(df$lx)
  df$ly = as.numeric(df$ly)
  df
}


for(ii in 1:length(ls)){
  mat=gen_mos(ls[[dd[ii,i]]][1])
  if (nrow(mat) >=2) {
    mat_ref <- domap(mat)
    tt <- matrix(nrow=max(mat_ref$ly),ncol = max(mat_ref$lx))
    tot <- nrow(mat_ref)
    for(k in 1:tot){
      tt[mat_ref$ly[k],mat_ref$lx[k]]=as.character(mat_ref$image)[k]
    }
    write.table(tt,paste0(output_path, "mosaic_",tot,"_",ii,".csv"), row.names = F, col.names = F, sep=',', na='')
  }
}