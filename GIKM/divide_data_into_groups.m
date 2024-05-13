function [group] = divide_data_into_groups(N,nGroup,random_flag)
nElementPerGroup = floor(N/nGroup);
group = repelem((1:nGroup),nElementPerGroup);
if numel(group) < N
    group = [group nGroup*ones(1,N-numel(group))];
end
if random_flag
    group = group(randperm(numel(group)));
end
return