import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import subprocess
import re
from sympy.printing.octave import octave_code
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Aproximação Polinomial - CEFET MG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS simples para estilização das tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 6px 6px 0px 0px;
        border: 1px solid #e9ecef;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 3px solid #104080;
        font-weight: bold;
        color: #104080;
    }
</style>
""", unsafe_allow_html=True)

# Título do Painel Interno
st.title("Painel de Cálculo: Aproximação Polinomial 📊")
st.markdown("Insira os parâmetros abaixo para calcular e comparar as aproximações polinomiais usando os algoritmos do Octave.")

# =========================================================================
# FORMULÁRIO DE CONFIGURAÇÕES CENTRALIZADO (Substituindo a Sidebar)
# =========================================================================
with st.container():
    st.subheader("⚙️ Parâmetros de Entrada")
    
    # Primeira linha de inputs: Fórmulas
    col_f, col_w = st.columns(2)
    with col_f:
        input_f_str = st.text_input("Função f(x):", value="1/x", help="Exemplo: sin(pi*x), 1/x, exp(x)")
    with col_w:
        input_w_str = st.text_input("Função Peso w(x):", value="1", help="Padrão w(x) = 1")
        
    # Segunda linha de inputs: Intervalos, Grau e Simpson
    col_a, col_b, col_n, col_m = st.columns(4)
    with col_a:
        val_a = st.number_input("Limite Inferior a:", value=1.0)
    with col_b:
        val_b = st.number_input("Limite Superior b:", value=3.0)
    with col_n:
        val_n = st.slider("Grau do Polinômio (n):", min_value=1, max_value=6, value=2)
    with col_m:
        val_m = st.number_input("Subintervalos Simpson (m - par):", min_value=2, step=2, value=1000)


import os
import shutil

x = sp.Symbol('x')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_octave_path():
    # 1. Tenta encontrar no PATH (funciona no Linux do Streamlit Cloud)
    path = shutil.which("octave")
    if path:
        return path
    # 2. Fallbacks comuns para macOS e outros caminhos
    fallbacks = [
        "/opt/homebrew/bin/octave",
        "/usr/local/bin/octave",
        "/usr/bin/octave"
    ]
    for p in fallbacks:
        if os.path.exists(p):
            return p
    return "octave"

# =========================================================================
# FUNÇÕES AUXILIARES E CHAMADAS DE SISTEMA
# =========================================================================

def parse_and_vectorize_for_octave(expr_str):
    expr_str = expr_str.lower()
    expr_str = expr_str.replace('\\pi', 'pi').replace('π', 'pi')
    expr_str = expr_str.replace('sen', 'sin')
    expr_str = expr_str.replace('tg', 'tan')
    expr_str = expr_str.replace('^', '**')
    
    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(expr_str, transformations=transformations)
    return expr, octave_code(expr)

def calcular_erros_aproximacao(poly_sympy, f_str, w_str, a, b):
    try:
        f_expr, _ = parse_and_vectorize_for_octave(f_str)
        w_expr, _ = parse_and_vectorize_for_octave(w_str)
        
        f_num = sp.lambdify(x, f_expr, 'numpy')
        w_num = sp.lambdify(x, w_expr, 'numpy')
        p_num = sp.lambdify(x, poly_sympy, 'numpy')
        
        x_err = np.linspace(a, b, 1001)
        y_real = f_num(x_err)
        if np.isscalar(y_real):
            y_real = np.full_like(x_err, y_real)
            
        y_poly = p_num(x_err)
        if np.isscalar(y_poly):
            y_poly = np.full_like(x_err, y_poly)
            
        w_val = w_num(x_err)
        if np.isscalar(w_val):
            w_val = np.full_like(x_err, w_val)
            
        # Erro máximo absoluto (L_inf)
        erro_max = np.max(np.abs(y_real - y_poly))
        
        # Erro quadrático médio integrado (L2) via regra do trapézio
        dx = (b - a) / 1000
        diff_sq = w_val * (y_real - y_poly)**2
        erro_l2 = (dx / 2) * (diff_sq[0] + 2 * np.sum(diff_sq[1:-1]) + diff_sq[-1])
        
        return erro_max, erro_l2
    except Exception:
        return None, None

def run_octave(script_str):
    octave_cmd = get_octave_path()
    result = subprocess.run(
        [octave_cmd, "--no-gui", "--no-window-system", "--eval", script_str],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def parse_vector(output_str, section_header):
    lines = output_str.split("\n")
    started = False
    numbers = []
    for line in lines:
        if section_header in line:
            started = True
            continue
        if started:
            if "===" in line:
                break
            matches = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", line)
            if matches:
                numbers.extend([float(num) for num in matches])
    return np.array(numbers)

def parse_cell_array(output_str, section_header):
    lines = output_str.split("\n")
    started = False
    polynomials = []
    current_poly = []
    for line in lines:
        if section_header in line:
            started = True
            continue
        if started:
            if "=== END_PHI ===" in line or "===" in line and not "PHI" in line:
                if current_poly:
                    polynomials.append(current_poly)
                break
            if "phi_" in line:
                if current_poly:
                    polynomials.append(current_poly)
                    current_poly = []
                continue
            matches = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", line)
            if matches:
                current_poly.extend([float(num) for num in matches])
    if current_poly and len(polynomials) == 0:
        polynomials.append(current_poly)
    return polynomials

def octave_minimos_quadrados(f_str, a, b, n, w_str):
    _, f_vec = parse_and_vectorize_for_octave(f_str)
    _, w_vec = parse_and_vectorize_for_octave(w_str)
    script = f"""
    f = @(x) {f_vec};
    w = @(x) {w_vec};
    [c, A, B] = minimos_quadrados(f, {a}, {b}, {n}, w);
    disp('=== COEF ===');
    disp(c');
    disp('=== MAT_A ===');
    disp(A);
    disp('=== VEC_B ===');
    disp(B);
    """
    stdout = run_octave(script)
    coefs = parse_vector(stdout, "=== COEF ===")
    a_flat = parse_vector(stdout, "=== MAT_A ===")
    mat_A = a_flat.reshape((n + 1, n + 1))
    vec_B = parse_vector(stdout, "=== VEC_B ===")
    poly_expr = sum(coefs[k] * (x**k) for k in range(len(coefs)))
    return coefs, sp.expand(poly_expr), mat_A, vec_B

def octave_gram_schmidt(f_str, a, b, n, w_str):
    _, f_vec = parse_and_vectorize_for_octave(f_str)
    _, w_vec = parse_and_vectorize_for_octave(w_str)
    script = f"""
    f = @(x) {f_vec};
    w = @(x) {w_vec};
    [p_coef, phi, alpha] = gram_schmidt(f, {a}, {b}, {n}, w);
    disp('=== P_COEF ===');
    disp(p_coef);
    disp('=== ALPHA ===');
    disp(alpha);
    disp('=== PHI ===');
    for k = 1:length(phi)
        fprintf('phi_%d\\n', k);
        disp(phi{{k}});
    end
    disp('=== END_PHI ===');
    """
    stdout = run_octave(script)
    p_coef = parse_vector(stdout, "=== P_COEF ===")
    alpha = parse_vector(stdout, "=== ALPHA ===")
    phi_raw = parse_cell_array(stdout, "=== PHI ===")
    poly_expr = sum(coef * (x**(len(p_coef) - 1 - i)) for i, coef in enumerate(p_coef))
    phi_exprs = []
    for p_vec in phi_raw:
        phi_exprs.append(sum(coef * (x**(len(p_vec) - 1 - i)) for i, coef in enumerate(p_vec)))
    return p_coef, sp.expand(poly_expr), phi_exprs, alpha

def octave_gram_schmidt_simpson(f_str, a, b, n, w_str, m):
    _, f_vec = parse_and_vectorize_for_octave(f_str)
    _, w_vec = parse_and_vectorize_for_octave(w_str)
    script = f"""
    f = @(x) {f_vec};
    w = @(x) {w_vec};
    [p_coef, phi, alpha] = gram_schmidt_simpson(f, {a}, {b}, {n}, w, {m});
    disp('=== P_COEF ===');
    disp(p_coef);
    disp('=== ALPHA ===');
    disp(alpha);
    disp('=== PHI ===');
    for k = 1:length(phi)
        fprintf('phi_%d\\n', k);
        disp(phi{{k}});
    end
    disp('=== END_PHI ===');
    """
    stdout = run_octave(script)
    p_coef = parse_vector(stdout, "=== P_COEF ===")
    alpha = parse_vector(stdout, "=== ALPHA ===")
    phi_raw = parse_cell_array(stdout, "=== PHI ===")
    poly_expr = sum(coef * (x**(len(p_coef) - 1 - i)) for i, coef in enumerate(p_coef))
    phi_exprs = []
    for p_vec in phi_raw:
        phi_exprs.append(sum(coef * (x**(len(p_vec) - 1 - i)) for i, coef in enumerate(p_vec)))
    return p_coef, sp.expand(poly_expr), phi_exprs, alpha

# =========================================================================
# PROCESSAMENTO E EXIBIÇÃO
# =========================================================================
try:
    coef_mq, poly_mq, mat_A_mq, vec_B_mq = octave_minimos_quadrados(input_f_str, val_a, val_b, val_n, input_w_str)
    p_coef_gs, poly_gs, phi_gs, alpha_gs = octave_gram_schmidt(input_f_str, val_a, val_b, val_n, input_w_str)
    p_coef_gss, poly_gss, phi_gss, alpha_gss = octave_gram_schmidt_simpson(input_f_str, val_a, val_b, val_n, input_w_str, val_m)
    
    st.divider()
    tab1, tab2, tab3 = st.tabs([
        "1. Mínimos Quadrados Contínuos", 
        "2. Gram-Schmidt (Integral)", 
        "3. Gram-Schmidt (Simpson)"
    ])
    
    # -----------------------------------------------------------------
    # ABA 1: MÍNIMOS QUADRADOS CONTÍNUOS
    # -----------------------------------------------------------------
    with tab1:
        st.header("Método dos Mínimos Quadrados Contínuos (Via Octave)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Polinômio Aproximador Calculado pelo Octave")
            st.latex(f"p_{{{val_n}}}(x) = {sp.latex(poly_mq)}")
            
            st.subheader("Sistema de Equações Normais Desenvolvido ($A c = B$)")
            sys_latex = r"\begin{bmatrix} "
            for row in mat_A_mq:
                sys_latex += " & ".join(f"{val:.6f}" for val in row) + r" \\ "
            sys_latex += r"\end{bmatrix} \begin{bmatrix} "
            for i in range(len(coef_mq)):
                sys_latex += f"c_{{{i}}} \\\\ "
            sys_latex += r"\end{bmatrix} = \begin{bmatrix} "
            for val in vec_B_mq:
                sys_latex += f"{val:.6f} \\\\ "
            sys_latex += r"\end{bmatrix}"
            st.latex(sys_latex)
            
            st.subheader("Coeficientes do Sistema ($c_i$ em ordem crescente de grau)")
            for i, c_val in enumerate(coef_mq):
                st.markdown(f"**$c_{{{i}}}$** = `{c_val:.10f}`")
        
        with col2:
            st.subheader("Código de Chamada no Octave")
            _, f_vec_oct = parse_and_vectorize_for_octave(input_f_str)
            _, w_vec_oct = parse_and_vectorize_for_octave(input_w_str)
            oct_code = f"""% Arquivo executado: minimos_quadrados.m
f = @(x) {f_vec_oct};
w = @(x) {w_vec_oct};
a = {val_a};
b = {val_b};
n = {val_n};

[c, A, B] = minimos_quadrados(f, a, b, n, w);
"""
            st.code(oct_code, language="matlab")
            
            # Cálculo e exibição do erro de aproximação
            err_max, err_l2 = calcular_erros_aproximacao(poly_mq, input_f_str, input_w_str, val_a, val_b)
            if err_max is not None:
                st.subheader("Erros de Aproximação")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Erro Máximo Absoluto ($L_\\infty$)", f"{err_max:.6e}")
                with c2:
                    st.metric("Erro Quadrático Ponderado ($L_2$)", f"{err_l2:.6e}")
        
        with st.expander("📄 Script Octave Completo (copie e execute) — minimos_quadrados"):
            st.info("💡 **Como rodar no Octave:**\n"
                    "Salve o código abaixo em um arquivo chamado `main_minimos_quadrados.m` e execute digitando `main_minimos_quadrados` no console do Octave.\n\n"
                    "Dessa forma, a função auxiliar de exibição e a função de cálculo rodam juntas e organizadas sem nenhum erro de compilação.")
            try:
                with open(PROJECT_DIR + "/minimos_quadrados.m", "r") as mf:
                    func_code = mf.read()
                _, f_oct = parse_and_vectorize_for_octave(input_f_str)
                _, w_oct = parse_and_vectorize_for_octave(input_w_str)
                script_completo = f"""function main_minimos_quadrados()
    % === SCRIPT DE EXECUÇÃO ===
    % SALVE ESTE ARQUIVO COMO: main_minimos_quadrados.m
    % Para rodar, digite 'main_minimos_quadrados' no console do Octave.

    % Parâmetros
    f = @(x) {f_oct};
    w = @(x) {w_oct};
    a = {val_a};
    b = {val_b};
    n = {val_n};

    % Chamada da função
    [c, A, B] = minimos_quadrados(f, a, b, n, w);

    % Exibição dos resultados
    disp('Coeficientes c:'); disp(c');
    disp('Matriz A:'); disp(A);
    disp('Vetor B:'); disp(B);
    disp('Polinômio aproximador obtido:');
    exibir_polinomio(c, 'asc');
end

% === FUNÇÕES DE SUPORTE ===
{func_code}"""
                st.code(script_completo, language="matlab")
            except Exception:
                st.warning("Arquivo minimos_quadrados.m não encontrado.")

    # -----------------------------------------------------------------
    # ABA 2: GRAM-SCHMIDT (INTEGRAL PADRÃO)
    # -----------------------------------------------------------------
    with tab2:
        st.header("Gram-Schmidt com Integração Adaptativa (Via Octave)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Polinômio Aproximador Ortogonal")
            st.latex(f"p_{{{val_n}}}(x) = {sp.latex(poly_gs)}")
            
            st.subheader("Polinômios Ortogonais Gerados $\\theta_k(x)$")
            for k, phi_k in enumerate(phi_gs):
                st.latex(f"\\theta_{{{k}}}(x) = {sp.latex(phi_k)}")
                
            st.subheader("Coeficientes de Regressão $\\alpha_k$")
            for k, alpha_k in enumerate(alpha_gs):
                st.markdown(f"**$\\alpha_{{{k}}}$** = `{alpha_k:.10f}`")
        
        with col2:
            st.subheader("Código de Chamada no Octave")
            _, f_vec_oct = parse_and_vectorize_for_octave(input_f_str)
            _, w_vec_oct = parse_and_vectorize_for_octave(input_w_str)
            oct_code_gs = f"""% Arquivo executado: gram_schmidt.m
f = @(x) {f_vec_oct};
w = @(x) {w_vec_oct};
a = {val_a};
b = {val_b};
n = {val_n};

[p_coef, phi, alpha] = gram_schmidt(f, a, b, n, w);
"""
            st.code(oct_code_gs, language="matlab")
            
            # Cálculo e exibição do erro de aproximação
            err_max, err_l2 = calcular_erros_aproximacao(poly_gs, input_f_str, input_w_str, val_a, val_b)
            if err_max is not None:
                st.subheader("Erros de Aproximação")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Erro Máximo Absoluto ($L_\\infty$)", f"{err_max:.6e}")
                with c2:
                    st.metric("Erro Quadrático Ponderado ($L_2$)", f"{err_l2:.6e}")
        
        with st.expander("📄 Script Octave Completo (copie e execute) — gram_schmidt"):
            st.info("💡 **Como rodar no Octave:**\n"
                    "Salve o código abaixo em um arquivo chamado `main_gram_schmidt.m` e execute digitando `main_gram_schmidt` no console do Octave.\n\n"
                    "Dessa forma, a função auxiliar de exibição e a função de cálculo rodam juntas e organizadas sem nenhum erro de compilação.")
            try:
                with open(PROJECT_DIR + "/gram_schmidt.m", "r") as mf:
                    func_code = mf.read()
                _, f_oct = parse_and_vectorize_for_octave(input_f_str)
                _, w_oct = parse_and_vectorize_for_octave(input_w_str)
                script_completo = f"""function main_gram_schmidt()
    % === SCRIPT DE EXECUÇÃO ===
    % SALVE ESTE ARQUIVO COMO: main_gram_schmidt.m
    % Para rodar, digite 'main_gram_schmidt' no console do Octave.

    % Parâmetros
    f = @(x) {f_oct};
    w = @(x) {w_oct};
    a = {val_a};
    b = {val_b};
    n = {val_n};

    % Chamada da função
    [p_coef, phi, alpha] = gram_schmidt(f, a, b, n, w);

    % Exibição dos resultados
    disp('Coeficientes do polinômio (grau decrescente):'); disp(p_coef);
    disp('Polinômio aproximador obtido:');
    exibir_polinomio(p_coef, 'desc');
    disp('Coeficientes alpha:'); disp(alpha);
    for k = 1:length(phi)
      fprintf('phi_%d: ', k-1); disp(phi{{k}});
    end
end

% === FUNÇÕES DE SUPORTE ===
{func_code}"""
                st.code(script_completo, language="matlab")
            except Exception:
                st.warning("Arquivo gram_schmidt.m não encontrado.")

    # -----------------------------------------------------------------
    # ABA 3: GRAM-SCHMIDT (SIMPSON COMPOSTO)
    # -----------------------------------------------------------------
    with tab3:
        st.header("Gram-Schmidt com Simpson Composto (Via Octave)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("Polinômio Aproximador Ortogonal (Simpson)")
            st.latex(f"p_{{{val_n}}}(x) = {sp.latex(poly_gss)}")
            
            st.subheader("Polinômios Ortogonais Gerados $\\theta_k(x)$")
            for k, phi_k in enumerate(phi_gss):
                st.latex(f"\\theta_{{{k}}}(x) = {sp.latex(phi_k)}")
                
            st.subheader("Coeficientes de Regressão $\\alpha_k$")
            for k, alpha_k in enumerate(alpha_gss):
                st.markdown(f"**$\\alpha_{{{k}}}$** = `{alpha_k:.10f}`")
        
        with col2:
            st.subheader("Código de Chamada no Octave")
            _, f_vec_oct = parse_and_vectorize_for_octave(input_f_str)
            _, w_vec_oct = parse_and_vectorize_for_octave(input_w_str)
            oct_code_gss = f"""% Arquivo executado: gram_schmidt_simpson.m
f = @(x) {f_vec_oct};
w = @(x) {w_vec_oct};
a = {val_a};
b = {val_b};
n = {val_n};
m = {val_m};

[p_coef, phi, alpha] = gram_schmidt_simpson(f, a, b, n, w, m);
"""
            st.code(oct_code_gss, language="matlab")
            
            # Cálculo e exibição do erro de aproximação
            err_max, err_l2 = calcular_erros_aproximacao(poly_gss, input_f_str, input_w_str, val_a, val_b)
            if err_max is not None:
                st.subheader("Erros de Aproximação")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Erro Máximo Absoluto ($L_\\infty$)", f"{err_max:.6e}")
                with c2:
                    st.metric("Erro Quadrático Ponderado ($L_2$)", f"{err_l2:.6e}")
        
        with st.expander("📄 Script Octave Completo (copie e execute) — gram_schmidt_simpson"):
            st.info("💡 **Como rodar no Octave:**\n"
                    "Salve o código abaixo em um arquivo chamado `main_gram_schmidt_simpson.m` e execute digitando `main_gram_schmidt_simpson` no console do Octave.\n\n"
                    "Dessa forma, a função auxiliar de exibição e a função de cálculo rodam juntas e organizadas sem nenhum erro de compilação.")
            try:
                with open(PROJECT_DIR + "/gram_schmidt_simpson.m", "r") as mf:
                    func_code = mf.read()
                _, f_oct = parse_and_vectorize_for_octave(input_f_str)
                _, w_oct = parse_and_vectorize_for_octave(input_w_str)
                script_completo = f"""function main_gram_schmidt_simpson()
    % === SCRIPT DE EXECUÇÃO ===
    % SALVE ESTE ARQUIVO COMO: main_gram_schmidt_simpson.m
    % Para rodar, digite 'main_gram_schmidt_simpson' no console do Octave.

    % Parâmetros
    f = @(x) {f_oct};
    w = @(x) {w_oct};
    a = {val_a};
    b = {val_b};
    n = {val_n};
    m = {val_m};

    % Chamada da função
    [p_coef, phi, alpha] = gram_schmidt_simpson(f, a, b, n, w, m);

    % Exibição dos resultados
    disp('Coeficientes do polinômio (grau decrescente):'); disp(p_coef);
    disp('Polinômio aproximador obtido:');
    exibir_polinomio(p_coef, 'desc');
    disp('Coeficientes alpha:'); disp(alpha);
    for k = 1:length(phi)
      fprintf('phi_%d: ', k-1); disp(phi{{k}});
    end
end

% === FUNÇÕES DE SUPORTE ===
{func_code}"""
                st.code(script_completo, language="matlab")
            except Exception:
                st.warning("Arquivo gram_schmidt_simpson.m não encontrado.")

    # -----------------------------------------------------------------
    # GRÁFICO COMPARATIVO
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("📊 Gráfico Comparativo Dinâmico")
    
    x_plot = np.linspace(val_a, val_b, 500)
    
    try:
        f_expr, _ = parse_and_vectorize_for_octave(input_f_str)
        f_num = sp.lambdify(x, f_expr, 'numpy')
        y_real = f_num(x_plot)
        if np.isscalar(y_real):
            y_real = np.full_like(x_plot, y_real)
            
        y_mq = sp.lambdify(x, poly_mq, 'numpy')(x_plot)
        if np.isscalar(y_mq):
            y_mq = np.full_like(x_plot, y_mq)
            
        y_gs = sp.lambdify(x, poly_gs, 'numpy')(x_plot)
        if np.isscalar(y_gs):
            y_gs = np.full_like(x_plot, y_gs)
            
        y_gss = sp.lambdify(x, poly_gss, 'numpy')(x_plot)
        if np.isscalar(y_gss):
            y_gss = np.full_like(x_plot, y_gss)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_plot, y_real, label="f(x) Original", color="#1f77b4", linewidth=2.5)
        ax.plot(x_plot, y_mq, label="Aproximação MQ (Octave)", color="#ff7f0e", linestyle="--", linewidth=2.0)
        ax.plot(x_plot, y_gss, label="Aproximação GS Simpson (Octave)", color="#d62728", linestyle="-.", linewidth=2.5)
        ax.plot(x_plot, y_gs, label="Aproximação GS Integral (Octave)", color="#2ca02c", linestyle=(0, (2, 3)), linewidth=2.0)
        
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_title("Comparação Visual das Aproximações Polinomiais", fontsize=12, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best")
        
        st.pyplot(fig)
    except Exception as e_plot:
        st.warning(f"Erro ao plotar as curvas no gráfico: {e_plot}")

    # =========================================================================
    # PAINEL DE CUSTO COMPUTACIONAL (Normalizado)
    # =========================================================================
    st.divider()
    st.subheader("⚡ Custo Computacional Comparativo")
    st.caption("Todos os custos são normalizados para **avaliações estimadas de f(x)** como parâmetro comum. "
               "A função `integral()` do Octave (quadratura adaptativa) usa ~150 avaliações por chamada. "
               "O Simpson Composto usa exatamente $(m+1)$ avaliações por integral.")
    
    # Estimativa: integral() adaptativa ≈ 150 avaliações por chamada
    AVALS_POR_INTEGRAL = 150
    
    integrais_mq = (val_n + 1) * (val_n + 2)
    integrais_gs = 6 * val_n
    integrais_gss = 6 * val_n
    
    custo_mq = integrais_mq * AVALS_POR_INTEGRAL + (val_n + 1)**3  # + resolução O(n³)
    custo_gs = integrais_gs * AVALS_POR_INTEGRAL
    custo_gss = integrais_gss * (val_m + 1)
    
    custos = {"Mínimos Quadrados": custo_mq, "Gram-Schmidt (Integral)": custo_gs, "Gram-Schmidt (Simpson)": custo_gss}
    mais_eficiente = min(custos, key=custos.get)
    
    def fmt(v):
        return f"{v:,.0f}".replace(",", ".")
    
    col_mq_c, col_gs_c, col_gss_c = st.columns(3)
    
    with col_mq_c:
        st.markdown("**1. Mínimos Quadrados**")
        st.metric("Integrais", f"{integrais_mq}")
        st.metric("Avaliações estimadas de f(x)", fmt(custo_mq))
        st.caption(f"$(n+1)(n+2) \\times 150 + (n+1)^3 = {fmt(custo_mq)}$")
        if mais_eficiente == "Mínimos Quadrados":
            st.success("✅ Mais eficiente")
    
    with col_gs_c:
        st.markdown("**2. Gram-Schmidt (Integral)**")
        st.metric("Integrais", f"{integrais_gs}")
        st.metric("Avaliações estimadas de f(x)", fmt(custo_gs))
        st.caption(f"$6n \\times 150 = {fmt(custo_gs)}$")
        if mais_eficiente == "Gram-Schmidt (Integral)":
            st.success("✅ Mais eficiente")
    
    with col_gss_c:
        st.markdown("**3. Gram-Schmidt (Simpson)**")
        st.metric("Integrais", f"{integrais_gss}")
        st.metric("Avaliações estimadas de f(x)", fmt(custo_gss))
        st.caption(f"$6n \\times (m+1) = {fmt(custo_gss)}$")
        if mais_eficiente == "Gram-Schmidt (Simpson)":
            st.success("✅ Mais eficiente")

except Exception as e:
    st.error(f"Erro ao executar os códigos do Octave. Detalhes: {e}")
