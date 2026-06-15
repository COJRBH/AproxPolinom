% =========================================================================
% PROCESSO DE GRAM-SCHMIDT - Teorema 8.7
% Construção de Polinômios Ortogonais em [a, b] com peso w(x)
%
% Referência: Seção 8.2 - Polinômios Ortogonais
%
% Método: Utiliza function handles (funções anônimas) para avaliação
%         numérica e integração via quadratura de Gauss-Legendre.
% =========================================================================

clear; clc; format long;

% =========================================================================
% 1. DEFINIÇÃO DAS ENTRADAS (altere aqui conforme necessário)
% =========================================================================

a = -1;          % Limite inferior do intervalo
b =  1;          % Limite superior do intervalo
n =  4;          % Grau máximo dos polinômios a serem gerados

% Função peso w(x)
% Para Polinômios de Legendre: w(x) = 1 em [-1, 1]
% Outros exemplos comuns:
%   Chebyshev tipo I : w = @(x) 1./sqrt(1 - x.^2)  em (-1,1)
%   Laguerre         : w = @(x) exp(-x)             em [0, Inf)
w = @(x) ones(size(x));   % w(x) = 1  (Legendre)

% =========================================================================
% 2. FUNÇÃO AUXILIAR DE INTEGRAÇÃO (Quadratura Gaussiana adaptativa)
% =========================================================================
% Usa a função built-in 'quadgk' do Octave para integração numérica
% adaptativa de alta precisão.

integra = @(f) quadgk(f, a, b, 'AbsTol', 1e-12, 'RelTol', 1e-10);

% =========================================================================
% 3. INICIALIZAÇÃO - GRAUS 0 e 1
% =========================================================================

fprintf('=================================================================\n');
fprintf('  PROCESSO DE GRAM-SCHMIDT - Polinômios Ortogonais\n');
fprintf('  Intervalo: [%g, %g]  |  Função peso: w(x)  |  Grau máximo: %d\n', a, b, n);
fprintf('=================================================================\n\n');

% Pré-aloca a célula de function handles para os polinômios
phi = cell(1, n + 1);   % phi{k+1} representa phi_k (índice 1-based)

% --- Grau 0: phi_0(x) = 1 ---
phi{1} = @(x) ones(size(x));
fprintf('phi_0(x) = 1\n');

if n == 0
    mostrar_resultados(phi, n);
    return;
end

% --- Grau 1: phi_1(x) = x - B1 ---
% B1 = integral(x * w(x) * phi_0(x)^2) / integral(w(x) * phi_0(x)^2)

num_B1 = integra(@(x) x .* w(x) .* phi{1}(x).^2);
den_B1 = integra(@(x)      w(x) .* phi{1}(x).^2);
B1 = num_B1 / den_B1;

phi{2} = @(x) (x - B1);
fprintf('B_1 = %.10f\n', B1);
fprintf('phi_1(x) = x - (%.10f)\n\n', B1);

% =========================================================================
% 4. LOOP RECURSIVO - GRAUS k = 2 até n
% =========================================================================
% Relação de recorrência (Teorema 8.7):
%   phi_k(x) = (x - B_k) * phi_{k-1}(x)  -  C_k * phi_{k-2}(x)
%
% Onde:
%   B_k = integral(x * w(x) * phi_{k-1}^2) / integral(w(x) * phi_{k-1}^2)
%   C_k = integral(x * w(x) * phi_{k-1} * phi_{k-2}) / integral(w(x) * phi_{k-2}^2)

B = zeros(1, n + 1);   % Armazena as constantes B_k
C = zeros(1, n + 1);   % Armazena as constantes C_k
B(1) = 0;              % B_0 não é usado (convenção)
B(2) = B1;

for k = 2:n
    phi_k1 = phi{k};       % phi_{k-1}  (índice k no array 1-based)
    phi_k2 = phi{k - 1};   % phi_{k-2}  (índice k-1 no array 1-based)

    % --- Calcula B_k ---
    num_Bk = integra(@(x) x .* w(x) .* phi_k1(x).^2);
    den_Bk = integra(@(x)      w(x) .* phi_k1(x).^2);
    Bk = num_Bk / den_Bk;

    % --- Calcula C_k ---
    num_Ck = integra(@(x) x .* w(x) .* phi_k1(x) .* phi_k2(x));
    den_Ck = integra(@(x)      w(x) .* phi_k2(x).^2);
    Ck = num_Ck / den_Ck;

    % --- Define phi_k usando a relação de recorrência ---
    % IMPORTANTE: captura as variáveis locais no closure para evitar
    % referência a variáveis que mudam no próximo loop.
    phi{k + 1} = make_phi(phi_k1, phi_k2, Bk, Ck);

    % Armazena as constantes para exibição
    B(k + 1) = Bk;
    C(k + 1) = Ck;

    fprintf('--- Grau k = %d ---\n', k);
    fprintf('  B_%d = %.10f\n', k, Bk);
    fprintf('  C_%d = %.10f\n', k, Ck);
    fprintf('  phi_%d(x) = (x - %.10f)*phi_%d(x) - (%.10f)*phi_%d(x)\n\n', ...
            k, Bk, k-1, Ck, k-2);
end

% =========================================================================
% 5. VERIFICAÇÃO DE ORTOGONALIDADE
% =========================================================================
fprintf('=================================================================\n');
fprintf('  VERIFICAÇÃO DE ORTOGONALIDADE\n');
fprintf('  <phi_i, phi_j> = integral(w(x)*phi_i(x)*phi_j(x), a, b)\n');
fprintf('  Deve ser ≈ 0 para i ≠ j\n');
fprintf('=================================================================\n\n');

tol_ortog = 1e-8;
is_ortogonal = true;

for i = 0:n
    for j = i+1:n
        prod_interno = integra(@(x) w(x) .* phi{i+1}(x) .* phi{j+1}(x));
        if abs(prod_interno) > tol_ortog
            fprintf('  FALHA: <phi_%d, phi_%d> = %.2e  (maior que tolerância!)\n', ...
                    i, j, prod_interno);
            is_ortogonal = false;
        end
    end
end

if is_ortogonal
    fprintf('  ✓ Todos os pares (i≠j) têm produto interno ≈ 0.\n');
    fprintf('  Os polinômios são ORTOGONAIS em [%g, %g] com a peso w(x).\n\n', a, b);
end

% =========================================================================
% 6. AVALIAÇÃO E COMPARAÇÃO COM POLINÔMIOS DE LEGENDRE (para w=1, [-1,1])
% =========================================================================
fprintf('=================================================================\n');
fprintf('  AVALIAÇÃO DOS POLINÔMIOS EM PONTOS DE TESTE\n');
fprintf('=================================================================\n');

x_teste = [-1, -0.5, 0, 0.5, 1];
fprintf('\n%-6s', 'x');
for k = 0:n
    fprintf('  phi_%d(x)       ', k);
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 6 + (n+1)*17));

for xi = x_teste
    fprintf('%-6.2f', xi);
    for k = 0:n
        fprintf('  %+.10f', phi{k+1}(xi));
    end
    fprintf('\n');
end

% =========================================================================
% 7. GRÁFICO DOS POLINÔMIOS
% =========================================================================
fprintf('\n');
x_plot = linspace(a, b, 500);
figure('Name', 'Polinômios Ortogonais - Gram-Schmidt', 'NumberTitle', 'off');
hold on; grid on;

cores = {'b', 'r', 'g', 'm', 'c', 'k', 'y'};
for k = 0:n
    y_plot = phi{k+1}(x_plot);
    plot(x_plot, y_plot, cores{mod(k, length(cores)) + 1}, ...
         'LineWidth', 2, 'DisplayName', sprintf('\\phi_%d(x)', k));
end

xlabel('x', 'FontSize', 12);
ylabel('\phi_k(x)', 'FontSize', 12);
title(sprintf('Polinômios Ortogonais (Gram-Schmidt) em [%g, %g]', a, b), 'FontSize', 14);
legend('show', 'Location', 'best', 'FontSize', 11);
yline(0, 'k--', 'LineWidth', 0.8);
hold off;

fprintf('Gráfico gerado com os %d polinômios ortogonais.\n', n + 1);
fprintf('=================================================================\n');

% =========================================================================
% FUNÇÃO AUXILIAR: cria o closure correto para phi_k
% (evita o problema de captura tardia de variáveis no loop)
% =========================================================================
function f = make_phi(phi_prev, phi_prev2, Bk, Ck)
    f = @(x) (x - Bk) .* phi_prev(x) - Ck .* phi_prev2(x);
end
