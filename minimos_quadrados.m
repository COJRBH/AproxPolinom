function [c, A, B] = minimos_quadrados(f, a, b, n, w)
  % Se a função peso w não for especificada, o padrão é w(x) = 1
  if nargin < 5
    w = @(x) ones(size(x));
  end

  A = zeros(n+1, n+1);
  B = zeros(n+1, 1);

  for i = 0:n
    for j = 0:n
      % A_{i+1, j+1} = integral( w(x) * x^(i+j) )
      A(i+1, j+1) = integral(@(x) w(x) .* x.^(i+j), a, b);
    end
    % B_{i+1} = integral( w(x) * f(x) * x^i )
    B(i+1) = integral(@(x) w(x) .* f(x) .* x.^i, a, b);
  end

  % Resolve o sistema linear A * c = B
  % Retorna os coeficientes c do polinômio: p(x) = c_1 + c_2*x + ... + c_{n+1}*x^n
  c = A \ B;
end

function exibir_polinomio(coefs, ordem)
    % ordem: 'desc' (padrão) para grau decrescente, 'asc' para crescente
    if nargin < 2
        ordem = 'desc';
    end
    if strcmp(ordem, 'asc')
        coefs = coefs(end:-1:1);
    end
    
    n = length(coefs) - 1;
    str = '';
    for i = 1:length(coefs)
        c = coefs(i);
        deg = n - i + 1;
        if c == 0
            continue;
        end
        
        % Sinal
        if c > 0
            if ~isempty(str)
                str = [str, ' + '];
            end
            val = c;
        else
            if ~isempty(str)
                str = [str, ' - '];
            else
                str = [str, '-'];
            end
            val = -c;
        end
        
        % Coeficiente
        if abs(val - 1) < 1e-9 && deg > 0
            coef_str = '';
        else
            coef_str = sprintf('%.4f', val);
        end
        
        % Variável e grau
        if deg == 0
            str = [str, coef_str];
        elseif deg == 1
            str = [str, coef_str, 'x'];
        else
            str = [str, coef_str, 'x^', num2str(deg)];
        end
    end
    if isempty(str)
        str = '0';
    end
    fprintf('p(x) = %s\n', str);
end
