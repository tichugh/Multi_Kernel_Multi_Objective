function MatingPool = F_mating(Population,FrontValue,CrowdDistance,N)


    [~,D] = size(Population);
    if size(Population,1) <N
        ss = N - size(Population,1);
        pop_temp = Population(1:ss,:);
        Population = [Population;pop_temp];
    end
    

    MatingPool = zeros(N,D);
    Rank = randperm(N);
    Pointer = 1;
    for i = 1 : 2 : N

        k = zeros(1,2);
        for j = 1 : 2
            if Pointer >= N
                Rank = randperm(N);
                Pointer = 1;
            end
            p = Rank(Pointer);
            q = Rank(Pointer+1);
            if FrontValue(p) < FrontValue(q)
                k(j) = p;
            elseif FrontValue(p) > FrontValue(q)
                k(j) = q;
            elseif CrowdDistance(p) > CrowdDistance(q)
                k(j) = p;
            else
                k(j) = q;
            end
            Pointer = Pointer+2;
        end
        MatingPool(i,:) = Population(k(1),:);
        MatingPool(i+1,:) = Population(k(2),:);
    end
end

