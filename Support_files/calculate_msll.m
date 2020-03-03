function msll = calculate_msll(y,y_pred,std_pred,sigman)
    sigman = 0;
    N = size(y,1);
    std_pred_ = (std_pred).^2 + (sigman).^2;
    msll = zeros(N,1);
    for i = 1:N
        msll(i,:) = 0.5*log(2*pi*std_pred_(i)) + (((y(i) - y_pred(i))^2)/(2*std_pred_(i)));
    end
   msll = mean(msll);
end