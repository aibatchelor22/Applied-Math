%load('svdModesSameGenre.mat')
totalConfusionMatrix=zeros(3);
for numberOfTrials = 1:20
    q1=randperm(30);
    q2=randperm(30);
    q3=randperm(30);
    xGroup1=v(1:30,1:50);
    xGroup2=v(31:60,1:50);
    xGroup3=v(61:90,1:50);
    xtrain=[xGroup1(q1(1:25),:); xGroup2(q2(1:25),:); xGroup3(q3(1:25),:)];
    xtest=[xGroup1(q1(26:end),:); xGroup2(q2(26:end),:); xGroup3(q3(26:end),:)];
    ctrain=[ones(25,1); 2*ones(25,1);3*ones(25,1)];
    actual=[ones(5,1);2*ones(5,1);3*ones(5,1)];
    %Apply Naive Bayes Classifier and test the model
    model=fitcecoc(xtrain,ctrain)
    pre=model.predict(xtest);
    %Create the current confusion matrix
    currentConfusionMatrix = confusionmat(actual,pre);
    %Update the total confusion matrix
    totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;
end