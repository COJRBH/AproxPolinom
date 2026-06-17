function [p_coef, phi, alpha] = gram_schmidt_simpson(f, a, b, n, w, m)
    % Se a função peso w não for especificada, o padrão é w(x) = 1
    if nargin < 5 || isempty(w)
        w = @(x) ones(size(x));
    end

    % Se o número de subintervalos m não for especificado, o padrão é 1000
    if nargin < 6 || isempty(m)
        m = 1000;
    end

    % Inicialização da célula de polinômios (vetores de coeficientes, grau decrescente)
    phi = cell(1, n + 1);
    phi{1} = [1]; % phi_0(x) = 1

    B = zeros(1, n);
    C = zeros(1, n);

    if n >= 1
        % Cálculo de B1
        num_B1 = simpson_composto(@(x) x .* w(x) .* polyval(phi{1}, x).^2, a, b, m);
        den_B1 = simpson_composto(@(x) w(x) .* polyval(phi{1}, x).^2, a, b, m);
        B(1) = num_B1 / den_B1;

        % Construção de phi_1(x) = (x - B1) * phi_0(x) -> Vetor [1, -B1]
        phi{2} = [1, -B(1)];
    end

    % Processo de recorrência para graus superiores (k >= 2)
    for k = 2:n
        % phi{k} represents phi_{k-1}
        num_Bk = simpson_composto(@(x) x .* w(x) .* polyval(phi{k}, x).^2, a, b, m);
        den_Bk = simpson_composto(@(x) w(x) .* polyval(phi{k}, x).^2, a, b, m);
        B(k) = num_Bk / den_Bk;

        num_Ck = simpson_composto(@(x) x .* w(x) .* polyval(phi{k}, x) .* polyval(phi{k-1}, x), a, b, m);
        den_Ck = simpson_composto(@(x) w(x) .* polyval(phi{k-1}, x).^2, a, b, m);
        C(k) = num_Ck / den_Ck;

        % Operação polinomial: (x - Bk)*phi_{k-1}(x) - Ck*phi_{k-2}(x)
        termo1 = conv([1, -B(k)], phi{k});
        termo2 = C(k) * phi{k-1};

        % Alinhamento dos comprimentos dos vetores para subtração
        len1 = length(termo1);
        len2 = length(termo2);
        termo2_pad = [zeros(1, len1 - len2), termo2];

        phi{k+1} = termo1 - termo2_pad;
    end

    % Cálculo dos coeficientes de regressão alpha_k e montagem do polinômio final
    p_coef = zeros(1, n + 1);
    alpha = zeros(1, n + 1);

    for k = 0:n
        num_alpha = simpson_composto(@(x) w(x) .* f(x) .* polyval(phi{k+1}, x), a, b, m);
        den_alpha = simpson_composto(@(x) w(x) .* polyval(phi{k+1}, x).^2, a, b, m);
        alpha(k+1) = num_alpha / den_alpha;

        % Acumulação do termo no polinômio final p_n(x)
        termo_p = alpha(k+1) * phi{k+1};
        len_p = length(p_coef);
        len_t = length(termo_p);
        termo_p_pad = [zeros(1, len_p - len_t), termo_p];
        p_coef = p_coef + termo_p_pad;
    end
end

function I = simpson_composto(g, a, b, N)
    % Validação de N (deve ser par para Simpson)
    if mod(N, 2) ~= 0
        N = N + 1;
    end

    h = (b - a) / N;
    
    XI0 = g(a) + g(b);
    XI1 = 0; % Soma de g(x_i) para i ímpar
    XI2 = 0; % Soma de g(x_i) para i par

    for i = 1:(N-1)
        x_i = a + i * h;
        g_xi = g(x_i);
        if mod(i, 2) == 0
            XI2 = XI2 + g_xi;
        else
            XI1 = XI1 + g_xi;
        end
    end

    I = (h / 3) * (XI0 + 2 * XI2 + 4 * XI1);
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
