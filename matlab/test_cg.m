
function test_cg(dim)

    %init:
    A = sparse(dim, dim);
    rhs(dim) = 1.0;

    rhs(1) = 1.0;
    A(1,1) = 10.0;
    A(1,2) = -1;
    A(1,3) = -2;
    A(2,1) = -1;
    A(2,2) = 10;
    A(2,3) = -1;
    A(2,4) = -2;
    for i=3:dim-2
        A(i,i+2) = -2;
        A(i,i+1) = -1;
        A(i,i) = 10;
        A(i,i-1) = -1;
        A(i,i-2) = -2;
    end
    A(dim-1, dim-3) = -2.0;
    A(dim-1, dim-2) = -1.0;
    A(dim-1, dim-1) = 10.0;
    A(dim-1, dim) = -1;
    A(dim, dim-2) = -2;
    A(dim, dim-1) = -1;
    A(dim, dim) = 10;

    %one startup calculation:    
    test = pcg(A, rhs', 1e-14, 300);
    tic;
    for i=1:10
        test = pcg(A, rhs', 1e-14, 300);
    end
    temp = toc;
    disp('CPU time: '); temp
    test(1)
    
    %one startup calculation:    
    test = viennacl_cg(A, rhs);
    tic;
    for i=1:10
        test = viennacl_cg(A, rhs);
    end
    temp = toc;
    disp('GPU time: '); temp
    test(1)
    
end
