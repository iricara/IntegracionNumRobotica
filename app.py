##st.set_page_config(
  ##  page_title="Integración Numérica en Robótica",
   ## page_icon="🤖",
   ## layout="wide"
##)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import (
    cumulative_trapezoid,
    cumulative_simpson
)

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================

st.set_page_config(
    page_title="Integración Numérica en Robótica",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# FUNCIONES
# =====================================================

def generar_trayectoria(tipo, t):

    if tipo == "Figura 8":

        x = 4*np.sin(t)
        y = 4*np.sin(2*t)

    elif tipo == "Circunferencia":

        x = 4*np.cos(t)
        y = 4*np.sin(t)

    elif tipo == "Espiral":

        r = 0.4*t

        x = r*np.cos(2*t)
        y = r*np.sin(2*t)

    elif tipo == "Lissajous":

        x = 4*np.sin(t)
        y = 4*np.sin(3*t)

    return x, y


def integrar_rect(signal, dt):

    salida = np.zeros_like(signal)

    for i in range(1, len(signal)):
        salida[i] = salida[i - 1] + signal[i - 1] * dt

    return salida


def integrar_richardson(signal, t, valor_inicial):

    ih2 = cumulative_trapezoid(
        signal,
        t,
        initial=0
    )

    t_h = t[::2]

    signal_h = signal[::2]

    ih = cumulative_trapezoid(
        signal_h,
        t_h,
        initial=0
    )

    ih_interp = np.interp(
        t,
        t_h,
        ih
    )

    resultado = (
        4 * ih2
        - ih_interp
    ) / 3

    resultado += valor_inicial

    return resultado


def ejecutar_simulacion(
    metodo,
    trayectoria,
    ruido
):

    T = 10
    fs = 10

    dt = 1 / fs

    t = np.arange(
        0,
        T,
        dt
    )

    np.random.seed(1)

    # ====================================
    # Trayectoria real
    # ====================================

    x_real, y_real = generar_trayectoria(
        trayectoria,
        t
    )

    vx_real = np.gradient(
        x_real,
        dt
    )

    vy_real = np.gradient(
        y_real,
        dt
    )

    ax_real = np.gradient(
        vx_real,
        dt
    )

    ay_real = np.gradient(
        vy_real,
        dt
    )

    # ====================================
    # Sensor con ruido
    # ====================================

    ax_sensor = (
        ax_real +
        ruido * np.random.randn(len(t))
    )

    ay_sensor = (
        ay_real +
        ruido * np.random.randn(len(t))
    )

    # ====================================
    # VELOCIDAD
    # ====================================

    if metodo == "Rectángulo":

        vx = integrar_rect(
            ax_sensor,
            dt
        )

        vy = integrar_rect(
            ay_sensor,
            dt
        )

        vx += vx_real[0]
        vy += vy_real[0]

    elif metodo == "Trapecio":

        vx = cumulative_trapezoid(
            ax_sensor,
            t,
            initial=0
        )

        vy = cumulative_trapezoid(
            ay_sensor,
            t,
            initial=0
        )

        vx += vx_real[0]
        vy += vy_real[0]

    elif metodo == "Simpson":

        vx = cumulative_simpson(
            ax_sensor,
            x=t,
            initial=0
        )

        vy = cumulative_simpson(
            ay_sensor,
            x=t,
            initial=0
        )

        vx += vx_real[0]
        vy += vy_real[0]

    elif metodo == "Richardson":

        vx = integrar_richardson(
            ax_sensor,
            t,
            vx_real[0]
        )

        vy = integrar_richardson(
            ay_sensor,
            t,
            vy_real[0]
        )

    # ====================================
    # POSICIÓN
    # ====================================

    if metodo == "Rectángulo":

        x = integrar_rect(
            vx,
            dt
        )

        y = integrar_rect(
            vy,
            dt
        )

        x += x_real[0]
        y += y_real[0]

    elif metodo == "Trapecio":

        x = cumulative_trapezoid(
            vx,
            t,
            initial=0
        )

        y = cumulative_trapezoid(
            vy,
            t,
            initial=0
        )

        x += x_real[0]
        y += y_real[0]

    elif metodo == "Simpson":

        x = cumulative_simpson(
            vx,
            x=t,
            initial=0
        )

        y = cumulative_simpson(
            vy,
            x=t,
            initial=0
        )

        x += x_real[0]
        y += y_real[0]

    elif metodo == "Richardson":

        x = integrar_richardson(
            vx,
            t,
            x_real[0]
        )

        y = integrar_richardson(
            vy,
            t,
            y_real[0]
        )

    # ====================================
    # Error final
    # ====================================

    error_final = np.sqrt(
        (x_real[-1] - x[-1])**2 +
        (y_real[-1] - y[-1])**2
    )

    # ====================================
    # Figura
    # ====================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        t,
        ax_real,
        label="Real",
        linewidth=2
    )

    axes[0].plot(
        t,
        ax_sensor,
        label="Sensor",
        alpha=0.6
    )

    axes[0].set_title(
        f"Aceleración (ruido={ruido:.2f})"
    )

    axes[0].legend()

    axes[1].plot(
        x_real,
        y_real,
        color="black",
        linewidth=3,
        label="Trayectoria real"
    )

    axes[1].plot(
        x,
        y,
        "--",
        linewidth=2,
        label=metodo
    )

    axes[1].scatter(
        x_real[0],
        y_real[0],
        color="green",
        s=80,
        label="Inicio"
    )

    axes[1].scatter(
        x_real[-1],
        y_real[-1],
        color="red",
        s=80,
        label="Fin"
    )

    axes[1].axis("equal")
    axes[1].grid(True)

    axes[1].set_title(
        "Trayectoria reconstruida"
    )

    axes[1].legend()

    plt.tight_layout()

    return fig, error_final


# =====================================================
# INTERFAZ
# =====================================================

st.title(
    "🤖 Integración Numérica en Robótica"
)

st.markdown("""
¿Cómo reconstruye un robot su movimiento si solo conoce las mediciones de sus sensores?

Esta simulación permite comparar distintos métodos de integración numérica para reconstruir una trayectoria a partir de datos de aceleración.
""")

with st.sidebar:

    st.header("Parámetros")

    metodo = st.radio(
        "Método",
        [
            "Rectángulo",
            "Trapecio",
            "Simpson",
            "Richardson"
        ]
    )

    trayectoria = st.selectbox(
        "Trayectoria",
        [
            "Figura 8",
            "Circunferencia",
            "Espiral",
            "Lissajous"
        ]
    )

    ruido = st.slider(
        "Ruido",
        0.0,
        1.0,
        0.5,
        0.05
    )

    generar = st.button(
        "🚀 Generar simulación"
    )

# =====================================================
# EJECUCIÓN
# =====================================================

if generar:

    fig, error_final = ejecutar_simulacion(
        metodo,
        trayectoria,
        ruido
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        st.pyplot(fig)

    with col2:

        st.metric(
            "Error Final",
            f"{error_final:.3f} m"
        )

        st.info(
            f"""
Método: {metodo}

Trayectoria: {trayectoria}

Ruido: {ruido:.2f}
"""
        )

        if metodo == "Richardson":

            st.success(
                """
Richardson combina dos aproximaciones
obtenidas mediante Trapecio para
reducir parte del error numérico.
"""
            )

        elif metodo == "Simpson":

            st.success(
                """
Simpson utiliza parábolas para
aproximar la curva con mayor precisión.
"""
            )

        elif metodo == "Trapecio":

            st.success(
                """
Trapecio considera la variación
entre dos mediciones consecutivas.
"""
            )

        else:

            st.success(
                """
Rectángulo asume que la señal
permanece constante en cada intervalo.
"""
            )
