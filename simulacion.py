import pygame
import sys
import math
import os
from collections import deque
from datetime import datetime
import control as ct
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.units import inch
from PIL import Image as PILImage

pygame.init()

ANCHO, ALTO = 1600, 900
FPS = 60

BG_DARK = (10, 12, 20)
BG_CARD = (20, 25, 40)
BG_CARD_LIGHT = (30, 38, 55)
BG_CARD_LIGHTER = (40, 48, 65)
ACCENT_BLUE = (66, 135, 245)
ACCENT_GREEN = (40, 205, 100)
ACCENT_RED = (245, 75, 75)
ACCENT_ORANGE = (255, 140, 30)
ACCENT_YELLOW = (255, 200, 20)
ACCENT_PURPLE = (168, 85, 247)
TEXT_PRIMARY = (250, 252, 255)
TEXT_SECONDARY = (155, 170, 190)
BORDER_COLOR = (60, 75, 100)

class ControladorPID:
    def __init__(self, kp=3.5, ki=0.25, kd=2.0, setpoint=75.0, 
                 output_limits=(30, 100), integrator_limits=(-50, 50)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min, self.output_max = output_limits
        self.int_min, self.int_max = integrator_limits

        self._prev_error = 0.0
        self._integral = 0.0
        self._prev_output = 30.0
        
        self.deadband = 1.0
        
        self.Kp = ct.TransferFunction([kp], [1])
        self.Ki = ct.TransferFunction([ki], [1, 0])
        self.Kd = ct.TransferFunction([kd, 0], [1])
        
        self.pid_system = self.Kp + self.Ki + self.Kd
        
        self.state = np.array([0.0])
        self.time_prev = 0.0

    def reiniciar(self):
        self._prev_error = 0.0
        self._integral = 0.0
        self._prev_output = 30.0
        self.state = np.array([0.0])

    def calcular(self, medida, dt):
        error = medida - self.setpoint
        
        if abs(error) < self.deadband and error < self._prev_error:
            self._prev_error = error
            return self._prev_output
        
        p = self.kp * error

        self._integral += error * dt
        self._integral = max(self.int_min, min(self.int_max, self._integral))
        i = self.ki * self._integral

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d = self.kd * derivative
        self._prev_error = error

        salida = p + i + d
        
        salida_final = self.output_min + salida
        
        salida_final = max(self.output_min, min(self.output_max, salida_final))
        
        if error > 10: 
            max_cambio = 25.0 * dt
        else:
            max_cambio = 15.0 * dt
        salida_final = self._prev_output + max(min(salida_final - self._prev_output, max_cambio), -max_cambio)
        
        self._prev_output = salida_final
        return salida_final

class ComputadoraSimulada:
    def __init__(self):
        # Temperaturas
        self.temperatura = 32.0
        self.temp_ambiente = 24.0

        # Carga y ventilador
        self.carga_cpu = 0.0
        self.velocidad_ventilador = 30.0

        self.tdp_max = 180.0
        self.thermal_capacity = 150.0
        self.natural_k = 0.3
        self.fan_k = 6.0
        
        self.fan_curve_exp = 1.2

        # Límites
        self.temp_critica = 95.0
        self.temp_maxima = 110.0
        self.temp_segura = 70.0
        self.temp_idle = 40.0

        self.dañada = False
        self.tiempo_sobrecalentamiento = 0.0

    def actualizar(self, velocidad_ventilador, dt=1/60):
        velocidad_ventilador = max(0.0, min(100.0, velocidad_ventilador))
        self.velocidad_ventilador = velocidad_ventilador

        carga_factor = (self.carga_cpu / 100.0)
        watts_generados = carga_factor * self.tdp_max

        delta_temp = max(0.01, self.temperatura - self.temp_ambiente)
        p_pasiva = self.natural_k * delta_temp
        
        fan_velocity_factor = (velocidad_ventilador / 100.0) ** self.fan_curve_exp
        p_activa = (self.fan_k * fan_velocity_factor) * delta_temp
        
        p_enfriamiento_total = p_pasiva + p_activa

        dTdt = (watts_generados - p_enfriamiento_total) / self.thermal_capacity
        self.temperatura += dTdt * dt

        if self.temperatura < self.temp_ambiente:
            self.temperatura = self.temp_ambiente

        if self.temperatura >= self.temp_maxima:
            self.temperatura = self.temp_maxima
            self.dañada = True

        if self.temperatura >= self.temp_critica:
            self.tiempo_sobrecalentamiento += dt
            if self.tiempo_sobrecalentamiento > 10.0:
                self.dañada = True
        else:
            self.tiempo_sobrecalentamiento = max(0.0, self.tiempo_sobrecalentamiento - dt * 2.0)

    def ajustar_carga(self, nueva_carga):
        self.carga_cpu = max(0.0, min(100.0, float(nueva_carga)))

    def esta_sobrecalentada(self):
        return self.temperatura >= self.temp_critica

    def obtener_estado_temperatura(self):
        if self.temperatura < self.temp_idle:
            return "IDLE", ACCENT_BLUE
        elif self.temperatura < self.temp_segura:
            return "NORMAL", ACCENT_GREEN
        elif self.temperatura < self.temp_critica:
            return "ALTA", ACCENT_ORANGE
        else:
            return "CRÍTICA", ACCENT_RED

class Boton:
    def __init__(self, x, y, ancho, alto, texto, color=ACCENT_BLUE):
        self.rect = pygame.Rect(x, y, ancho, alto)
        self.texto = texto
        self.color = color
        self.color_hover = tuple(min(c + 20, 255) for c in color)
        self.activo = False

    def dibujar_boton(self, screen, fuente):
        color = self.color_hover if self.activo else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        
        border_color = tuple(min(c + 30, 255) for c in color) if self.activo else BORDER_COLOR
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=10)

        texto_surf = fuente.render(self.texto, True, TEXT_PRIMARY)
        texto_rect = texto_surf.get_rect(center=self.rect.center)
        screen.blit(texto_surf, texto_rect)

    def contiene_punto(self, pos):
        return self.rect.collidepoint(pos)

class GeneradorReportes:
    def __init__(self):
        self.ruta_reportes = "reportes"
        if not os.path.exists(self.ruta_reportes):
            os.makedirs(self.ruta_reportes)
    
    def generar_reporte(self, computadora, pid, historial_temp, historial_vent, tiempo_total, pid_activado):
        """Genera un reporte PDF con los datos de la simulación"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"reporte_simulacion_{timestamp}.pdf"
        ruta_archivo = os.path.join(self.ruta_reportes, nombre_archivo)
        
        # Crear PDF
        doc = SimpleDocTemplate(ruta_archivo, pagesize=letter,
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Contenido del documento
        story = []
        styles = getSampleStyleSheet()
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4287f5'),
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("REPORTE DE SIMULACIÓN - SISTEMA DE CONTROL PID", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Información general
        fecha_reporte = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        info_general = f"""
        <b>Fecha del Reporte:</b> {fecha_reporte}<br/>
        <b>Duración de la Simulación:</b> {tiempo_total:.2f} segundos<br/>
        <b>Control PID:</b> {'ACTIVADO' if pid_activado else 'DESACTIVADO'}<br/>
        """
        story.append(Paragraph(info_general, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        if len(historial_temp) > 0:
            temps = list(historial_temp)
            temp_min = min(temps)
            temp_max = max(temps)
            temp_prom = sum(temps) / len(temps)
            
            stats_title = ParagraphStyle(
                'StatsTitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#28cd64'),
                spaceAfter=12
            )
            story.append(Paragraph("ESTADÍSTICAS DE TEMPERATURA", stats_title))
            
            temp_data = [
                ["Métrica", "Valor"],
                ["Temperatura Mínima", f"{temp_min:.1f}°C"],
                ["Temperatura Máxima", f"{temp_max:.1f}°C"],
                ["Temperatura Promedio", f"{temp_prom:.1f}°C"],
                ["Setpoint del PID", f"{pid.setpoint:.1f}°C"],
                ["Estado Final", f"{computadora.temperatura:.1f}°C"]
            ]
            
            temp_table = Table(temp_data, colWidths=[3*inch, 2.5*inch])
            temp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4287f5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#1e1f28')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3c4b64')),
            ]))
            story.append(temp_table)
            story.append(Spacer(1, 0.2*inch))
        
        if len(historial_vent) > 0:
            vents = list(historial_vent)
            vent_min = min(vents)
            vent_max = max(vents)
            vent_prom = sum(vents) / len(vents)
            
            vent_title = ParagraphStyle(
                'VentTitle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#a855f7'),
                spaceAfter=12
            )
            story.append(Paragraph("ESTADÍSTICAS DE VENTILADORES", vent_title))
            
            vent_data = [
                ["Métrica", "Valor"],
                ["Velocidad Mínima", f"{vent_min:.1f}%"],
                ["Velocidad Máxima", f"{vent_max:.1f}%"],
                ["Velocidad Promedio", f"{vent_prom:.1f}%"],
            ]
            
            vent_table = Table(vent_data, colWidths=[3*inch, 2.5*inch])
            vent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#a855f7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#1e1f28')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3c4b64')),
            ]))
            story.append(vent_table)
            story.append(Spacer(1, 0.2*inch))

        pid_title = ParagraphStyle(
            'PidTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#ff8c1e'),
            spaceAfter=12
        )
        story.append(Paragraph("PARÁMETROS DEL CONTROLADOR PID", pid_title))
        
        pid_data = [
            ["Parámetro", "Valor"],
            ["Ganancia Proporcional (Kp)", f"{pid.kp:.2f}"],
            ["Ganancia Integral (Ki)", f"{pid.ki:.2f}"],
            ["Ganancia Derivativa (Kd)", f"{pid.kd:.2f}"],
            ["Rango de Salida", f"{pid.output_min:.0f}% - {pid.output_max:.0f}%"],
            ["Rango Integrador", f"{pid.int_min:.0f} - {pid.int_max:.0f}"],
        ]
        
        pid_table = Table(pid_data, colWidths=[3*inch, 2.5*inch])
        pid_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff8c1e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#1e1f28')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3c4b64')),
        ]))
        story.append(pid_table)
        
        doc.build(story)
        return nombre_archivo, ruta_archivo

class Simulacion:
    def __init__(self):
        self.pantalla_completa = False
        self.screen = pygame.display.set_mode((ANCHO, ALTO), pygame.RESIZABLE | pygame.SHOWN)
        pygame.display.set_caption("Sistema de Control PID - Refrigeración CPU")
        self.clock = pygame.time.Clock()

        self.fuente_titulo = pygame.font.Font(None, 42)
        self.fuente_grande = pygame.font.Font(None, 56)
        self.fuente_media = pygame.font.Font(None, 32)
        self.fuente_pequena = pygame.font.Font(None, 24)
        self.fuente_mini = pygame.font.Font(None, 20)

        self.computadora = ComputadoraSimulada()
        self.pid = ControladorPID(kp=1.8, ki=0.12, kd=1.2, setpoint=75.0, 
                                   output_limits=(30, 100), integrator_limits=(-25, 25))
        self.pid_activado = False

        self.historial_temp = deque(maxlen=400)
        self.historial_ventilador = deque(maxlen=400)
        self.tiempo_total = 0.0
        
        # Generador de reportes
        self.generador_reportes = GeneradorReportes()
        self.mensaje_reporte = None
        self.tiempo_mensaje_reporte = 0.0

        self.angulo_ventilador1 = 0.0
        self.angulo_ventilador2 = 0.0

        self.btn_pid = Boton(50, 800, 200, 60, "PID: OFF", ACCENT_RED)
        self.btn_reiniciar = Boton(270, 800, 200, 60, "Reiniciar", BG_CARD_LIGHT)
        self.btn_reporte = Boton(1320, 800, 180, 60, "Generar Reporte", ACCENT_GREEN)

        self.btn_idle = Boton(550, 800, 140, 60, "Idle (5%)", ACCENT_BLUE)
        self.btn_oficina = Boton(710, 800, 160, 60, "Oficina (30%)", ACCENT_GREEN)
        self.btn_gaming = Boton(890, 800, 160, 60, "Gaming (70%)", ACCENT_ORANGE)
        self.btn_render = Boton(1070, 800, 180, 60, "Render (100%)", ACCENT_RED)

        self.corriendo = True

    def manejar_eventos(self):
        mouse_pos = pygame.mouse.get_pos()

        self.btn_pid.activo = self.btn_pid.contiene_punto(mouse_pos)
        self.btn_reiniciar.activo = self.btn_reiniciar.contiene_punto(mouse_pos)
        self.btn_reporte.activo = self.btn_reporte.contiene_punto(mouse_pos)
        self.btn_idle.activo = self.btn_idle.contiene_punto(mouse_pos)
        self.btn_oficina.activo = self.btn_oficina.contiene_punto(mouse_pos)
        self.btn_gaming.activo = self.btn_gaming.contiene_punto(mouse_pos)
        self.btn_render.activo = self.btn_render.contiene_punto(mouse_pos)

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.corriendo = False

            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_F11:
                    self.pantalla_completa = not self.pantalla_completa
                    if self.pantalla_completa:
                        self.screen = pygame.display.set_mode((ANCHO, ALTO), pygame.FULLSCREEN)
                    else:
                        self.screen = pygame.display.set_mode((ANCHO, ALTO), pygame.RESIZABLE)

            elif evento.type == pygame.MOUSEBUTTONDOWN:
                if self.btn_pid.contiene_punto(mouse_pos):
                    self.pid_activado = not self.pid_activado
                    self.btn_pid.texto = "PID: ON" if self.pid_activado else "PID: OFF"
                    self.btn_pid.color = ACCENT_GREEN if self.pid_activado else ACCENT_RED
                    if self.pid_activado:
                        self.pid.reiniciar()

                elif self.btn_reiniciar.contiene_punto(mouse_pos):
                    self.reiniciar()

                elif self.btn_reporte.contiene_punto(mouse_pos):
                    try:
                        nombre, ruta = self.generador_reportes.generar_reporte(
                            self.computadora, self.pid, self.historial_temp, 
                            self.historial_ventilador, self.tiempo_total, self.pid_activado
                        )
                        self.mensaje_reporte = f"Reporte generado: {nombre}"
                        self.tiempo_mensaje_reporte = 5.0
                        os.startfile(ruta)
                    except Exception as e:
                        self.mensaje_reporte = f"Error al generar reporte: {str(e)}"
                        self.tiempo_mensaje_reporte = 5.0

                elif self.btn_idle.contiene_punto(mouse_pos):
                    self.computadora.ajustar_carga(5)

                elif self.btn_oficina.contiene_punto(mouse_pos):
                    self.computadora.ajustar_carga(30)

                elif self.btn_gaming.contiene_punto(mouse_pos):
                    self.computadora.ajustar_carga(70)

                elif self.btn_render.contiene_punto(mouse_pos):
                    self.computadora.ajustar_carga(100)

    def reiniciar(self):
        self.computadora = ComputadoraSimulada()
        self.pid.reiniciar()
        self.historial_temp.clear()
        self.historial_ventilador.clear()
        self.tiempo_total = 0.0
        self.angulo_ventilador1 = 0.0
        self.angulo_ventilador2 = 0.0

    def actualizar(self, dt):
        if self.computadora.dañada:
            return

        if self.pid_activado:
            velocidad = self.pid.calcular(self.computadora.temperatura, dt)
        else:
            velocidad = 30.0

        self.computadora.actualizar(velocidad, dt)

        self.historial_temp.append(self.computadora.temperatura)
        self.historial_ventilador.append(self.computadora.velocidad_ventilador)

        velocidad_angular = self.computadora.velocidad_ventilador * 8.0
        self.angulo_ventilador1 = (self.angulo_ventilador1 + velocidad_angular * dt) % 360
        self.angulo_ventilador2 = (self.angulo_ventilador2 - velocidad_angular * dt) % 360

        self.tiempo_total += dt
        
        if self.tiempo_mensaje_reporte > 0:
            self.tiempo_mensaje_reporte -= dt

    def dibujar_ventilador(self, x, y, radio, angulo, color):
        pygame.draw.circle(self.screen, BORDER_COLOR, (x, y), radio + 5, 3)
        pygame.draw.circle(self.screen, BG_CARD, (x, y), radio)

        num_aspas = 6
        for i in range(num_aspas):
            ang = math.radians(angulo + i * 60)
            puntos = []
            for t in [0, 0.3, 0.6, 1.0]:
                r = radio * (0.2 + t * 0.7)
                offset = math.radians(25) * (1 - t)
                puntos.append((x + math.cos(ang + offset) * r, y + math.sin(ang + offset) * r))
            for t in [1.0, 0.6, 0.3, 0]:
                r = radio * (0.2 + t * 0.7)
                offset = math.radians(-25) * (1 - t)
                puntos.append((x + math.cos(ang + offset) * r, y + math.sin(ang + offset) * r))
            pygame.draw.polygon(self.screen, color, puntos)

        pygame.draw.circle(self.screen, BG_CARD_LIGHT, (x, y), radio * 0.25)
        pygame.draw.circle(self.screen, color, (x, y), radio * 0.2)

    def dibujar_cpu_chip(self):
        panel_x, panel_y = 50, 300
        panel_w, panel_h = 700, 450

        pygame.draw.rect(self.screen, BG_CARD, (panel_x, panel_y, panel_w, panel_h), border_radius=15)
        pygame.draw.rect(self.screen, ACCENT_PURPLE, (panel_x, panel_y, panel_w, panel_h), 3, border_radius=15)
        pygame.draw.rect(self.screen, BORDER_COLOR, (panel_x+2, panel_y+2, panel_w-4, panel_h-4), 1, border_radius=14)

        titulo = self.fuente_titulo.render("VISUALIZACIÓN CPU", True, TEXT_PRIMARY)
        self.screen.blit(titulo, (panel_x + 20, panel_y + 20))

        cpu_w, cpu_h = 200, 200
        cpu_x = panel_x + (panel_w - cpu_w) // 2
        cpu_y = panel_y + 120

        estado, color_temp = self.computadora.obtener_estado_temperatura()

        pygame.draw.rect(self.screen, BG_CARD_LIGHT, (cpu_x, cpu_y, cpu_w, cpu_h), border_radius=10)

        intensidad = min(1.0, (self.computadora.temperatura - 30) / 60)
        overlay_color = tuple(int(c * intensidad + BG_CARD_LIGHT[i] * (1 - intensidad))
                             for i, c in enumerate(color_temp))
        pygame.draw.rect(self.screen, overlay_color, (cpu_x + 10, cpu_y + 10, cpu_w - 20, cpu_h - 20), border_radius=8)
        pygame.draw.rect(self.screen, color_temp, (cpu_x, cpu_y, cpu_w, cpu_h), 3, border_radius=10)

        for i in range(8):
            y_pin = cpu_y + 30 + i * 20
            pygame.draw.rect(self.screen, BORDER_COLOR, (cpu_x - 15, y_pin, 10, 5))
            pygame.draw.rect(self.screen, BORDER_COLOR, (cpu_x + cpu_w + 5, y_pin, 10, 5))

        vent_color = ACCENT_GREEN
        if self.computadora.velocidad_ventilador > 50:
            vent_color = ACCENT_BLUE
        if self.computadora.velocidad_ventilador > 75:
            vent_color = ACCENT_ORANGE

        self.dibujar_ventilador(cpu_x - 100, cpu_y + cpu_h // 2, 60, self.angulo_ventilador1, vent_color)
        self.dibujar_ventilador(cpu_x + cpu_w + 100, cpu_y + cpu_h // 2, 60, self.angulo_ventilador2, vent_color)

        vel_text = f"{self.computadora.velocidad_ventilador:.0f}%"
        vel_surf = self.fuente_pequena.render(vel_text, True, TEXT_SECONDARY)
        self.screen.blit(vel_surf, (cpu_x - 135, cpu_y + cpu_h // 2 + 75))
        self.screen.blit(vel_surf, (cpu_x + cpu_w + 65, cpu_y + cpu_h // 2 + 75))

        rpm = int(self.computadora.velocidad_ventilador * 20)
        rpm_text = f"{rpm} RPM"
        rpm_surf = self.fuente_mini.render(rpm_text, True, TEXT_SECONDARY)
        self.screen.blit(rpm_surf, (cpu_x - 135, cpu_y + cpu_h // 2 + 95))
        self.screen.blit(rpm_surf, (cpu_x + cpu_w + 65, cpu_y + cpu_h // 2 + 95))

    def dibujar_panel_metricas(self):
        panel_x, panel_y = 50, 50
        panel_w, panel_h = 700, 220

        pygame.draw.rect(self.screen, BG_CARD, (panel_x, panel_y, panel_w, panel_h), border_radius=15)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (panel_x, panel_y, panel_w, panel_h), 3, border_radius=15)
        pygame.draw.rect(self.screen, BORDER_COLOR, (panel_x+2, panel_y+2, panel_w-4, panel_h-4), 1, border_radius=14)

        estado, color_estado = self.computadora.obtener_estado_temperatura()
        temp_text = f"{self.computadora.temperatura:.1f}°C"
        temp_surf = self.fuente_grande.render(temp_text, True, color_estado)
        self.screen.blit(temp_surf, (panel_x + 30, panel_y + 30))

        estado_surf = self.fuente_media.render(estado, True, color_estado)
        self.screen.blit(estado_surf, (panel_x + 30, panel_y + 95))

        barra_x, barra_y = panel_x + 30, panel_y + 145
        barra_w, barra_h = 640, 35

        pygame.draw.rect(self.screen, BG_CARD_LIGHT, (barra_x, barra_y, barra_w, barra_h), border_radius=8)

        proporcion = min(1.0, self.computadora.temperatura / 100)
        relleno_w = int(barra_w * proporcion)
        pygame.draw.rect(self.screen, color_estado, (barra_x, barra_y, relleno_w, barra_h), border_radius=8)

        marcadores = [(40, "40°", ACCENT_BLUE), (70, "70°", ACCENT_GREEN), (95, "95°", ACCENT_RED)]

        for temp_marc, label, color in marcadores:
            x_marc = barra_x + int((temp_marc / 100) * barra_w)
            pygame.draw.line(self.screen, color, (x_marc, barra_y - 5), (x_marc, barra_y + barra_h + 5), 2)
            label_surf = self.fuente_mini.render(label, True, TEXT_SECONDARY)
            self.screen.blit(label_surf, (x_marc - 15, barra_y + barra_h + 10))

        info_x, info_y = panel_x + 400, panel_y + 30

        labels = [
            ("Carga CPU:", f"{self.computadora.carga_cpu:.0f}%"),
            ("Ventiladores:", f"{self.computadora.velocidad_ventilador:.0f}%"),
            ("Control PID:", "ACTIVO" if self.pid_activado else "INACTIVO"),
            ("Setpoint:", f"{self.pid.setpoint:.0f}°C")
        ]

        for i, (label, valor) in enumerate(labels):
            y = info_y + i * 30
            label_surf = self.fuente_pequena.render(label, True, TEXT_SECONDARY)
            self.screen.blit(label_surf, (info_x, y))

            valor_surf = self.fuente_pequena.render(valor, True, TEXT_PRIMARY)
            self.screen.blit(valor_surf, (info_x + 160, y))

    def dibujar_grafica(self):
        panel_x, panel_y = 800, 50
        panel_w, panel_h = 750, 700

        pygame.draw.rect(self.screen, BG_CARD, (panel_x, panel_y, panel_w, panel_h), border_radius=15)
        pygame.draw.rect(self.screen, ACCENT_GREEN, (panel_x, panel_y, panel_w, panel_h), 3, border_radius=15)
        pygame.draw.rect(self.screen, BORDER_COLOR, (panel_x+2, panel_y+2, panel_w-4, panel_h-4), 1, border_radius=14)

        titulo = self.fuente_titulo.render("MONITOREO EN TIEMPO REAL", True, TEXT_PRIMARY)
        self.screen.blit(titulo, (panel_x + 20, panel_y + 20))

        graf_x, graf_y = panel_x + 50, panel_y + 100
        graf_w, graf_h = panel_w - 100, panel_h - 200

        pygame.draw.rect(self.screen, BG_CARD_LIGHT, (graf_x, graf_y, graf_w, graf_h), border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, (graf_x, graf_y, graf_w, graf_h), 1, border_radius=8)

        for i in range(6):
            y = graf_y + (i * graf_h // 5)
            pygame.draw.line(self.screen, BORDER_COLOR, (graf_x, y), (graf_x + graf_w, y), 1)
            temp_label = 100 - (i * 20)
            label_surf = self.fuente_mini.render(f"{temp_label}°C", True, TEXT_SECONDARY)
            self.screen.blit(label_surf, (graf_x - 45, y - 8))

        y_critica = graf_y + graf_h - int((95 / 100) * graf_h)
        y_segura = graf_y + graf_h - int((70 / 100) * graf_h)
        y_setpoint = graf_y + graf_h - int((self.pid.setpoint / 100) * graf_h)

        pygame.draw.line(self.screen, ACCENT_RED, (graf_x, y_critica), (graf_x + graf_w, y_critica), 2)
        zona_critica_txt = self.fuente_mini.render("95°C", True, ACCENT_RED)
        self.screen.blit(zona_critica_txt, (graf_x + graf_w + 10, y_critica - 8))

        pygame.draw.line(self.screen, ACCENT_ORANGE, (graf_x, y_segura), (graf_x + graf_w, y_segura), 2)
        zona_segura_txt = self.fuente_mini.render("70°C", True, ACCENT_ORANGE)
        self.screen.blit(zona_segura_txt, (graf_x + graf_w + 10, y_segura - 8))

        # Setpoint
        if self.pid_activado:
            pygame.draw.line(self.screen, ACCENT_YELLOW, (graf_x, y_setpoint), (graf_x + graf_w, y_setpoint), 2)
            setpoint_txt = self.fuente_mini.render(f"{self.pid.setpoint:.0f}°C", True, ACCENT_YELLOW)
            self.screen.blit(setpoint_txt, (graf_x + graf_w + 10, y_setpoint - 8))

        if len(self.historial_temp) > 1:
            puntos_temp, puntos_vent = [], []

            for i, temp in enumerate(self.historial_temp):
                x = graf_x + (i / max(1, len(self.historial_temp) - 1)) * graf_w
                y_temp = graf_y + graf_h - int((temp / 100) * graf_h)
                puntos_temp.append((x, max(graf_y, min(graf_y + graf_h, y_temp))))

                if i < len(self.historial_ventilador):
                    y_vent = graf_y + graf_h - int((self.historial_ventilador[i] / 100) * graf_h)
                    puntos_vent.append((x, max(graf_y, min(graf_y + graf_h, y_vent))))

            if len(puntos_vent) > 1:
                pygame.draw.lines(self.screen, ACCENT_BLUE, False, puntos_vent, 3)
            if len(puntos_temp) > 1:
                pygame.draw.lines(self.screen, ACCENT_RED, False, puntos_temp, 3)

        leyenda_y = panel_y + panel_h - 70
        leyenda_bg_h = 50
        leyenda_bg = pygame.Surface((panel_w - 40, leyenda_bg_h))
        leyenda_bg.set_alpha(50)
        leyenda_bg.fill(BG_CARD_LIGHT)
        self.screen.blit(leyenda_bg, (panel_x + 20, leyenda_y - 10))

        elementos_leyenda = [
            (ACCENT_RED, "Temperatura (°C)", graf_x),
            (ACCENT_BLUE, "Velocidad Ventiladores (%)", graf_x + 220),
        ]
        
        if self.pid_activado:
            elementos_leyenda.append((ACCENT_YELLOW, "Setpoint", graf_x + 500))

        for color, label, x_pos in elementos_leyenda:
            pygame.draw.line(self.screen, color, (x_pos, leyenda_y), (x_pos + 30, leyenda_y), 3)
            leg_surf = self.fuente_pequena.render(label, True, TEXT_PRIMARY)
            self.screen.blit(leg_surf, (x_pos + 40, leyenda_y - 8))

    def dibujar_advertencias(self):
        if self.computadora.dañada:
            overlay = pygame.Surface((ANCHO, ALTO))
            overlay.set_alpha(200)
            overlay.fill(ACCENT_RED)
            self.screen.blit(overlay, (0, 0))

            msg1 = self.fuente_grande.render("¡SISTEMA DAÑADO!", True, TEXT_PRIMARY)
            msg2 = self.fuente_media.render("La CPU se ha sobrecalentado", True, TEXT_PRIMARY)
            msg3 = self.fuente_pequena.render("Presiona 'Reiniciar' para comenzar de nuevo", True, TEXT_PRIMARY)

            self.screen.blit(msg1, (ANCHO // 2 - msg1.get_width() // 2, ALTO // 2 - 80))
            self.screen.blit(msg2, (ANCHO // 2 - msg2.get_width() // 2, ALTO // 2))
            self.screen.blit(msg3, (ANCHO // 2 - msg3.get_width() // 2, ALTO // 2 + 60))

        elif self.computadora.esta_sobrecalentada():
            if int(self.tiempo_total * 3) % 2 == 0:
                warning_surf = self.fuente_media.render("TEMPERATURA CRÍTICA", True, ACCENT_RED)
                self.screen.blit(warning_surf, (ANCHO // 2 - warning_surf.get_width() // 2, 770))

    def dibujar(self):
        self.screen.fill(BG_DARK)
        
        # Líneas decorativas en los bordes
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, 0), (ANCHO, 0), 2)
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, ALTO-1), (ANCHO, ALTO-1), 2)

        self.dibujar_panel_metricas()
        self.dibujar_cpu_chip()
        self.dibujar_grafica()

        self.btn_pid.dibujar_boton(self.screen, self.fuente_pequena)
        self.btn_reiniciar.dibujar_boton(self.screen, self.fuente_pequena)
        self.btn_reporte.dibujar_boton(self.screen, self.fuente_pequena)

        preset_label = self.fuente_pequena.render("Perfiles de Carga:", True, TEXT_SECONDARY)
        self.screen.blit(preset_label, (550, 775))

        self.btn_idle.dibujar_boton(self.screen, self.fuente_mini)
        self.btn_oficina.dibujar_boton(self.screen, self.fuente_mini)
        self.btn_gaming.dibujar_boton(self.screen, self.fuente_mini)
        self.btn_render.dibujar_boton(self.screen, self.fuente_mini)
        
        # Mostrar mensaje de reporte
        if self.tiempo_mensaje_reporte > 0 and self.mensaje_reporte:
            msg_surf = self.fuente_pequena.render(self.mensaje_reporte, True, ACCENT_GREEN)
            self.screen.blit(msg_surf, (ANCHO // 2 - msg_surf.get_width() // 2, 20))

        self.dibujar_advertencias()

        pygame.display.flip()

    def ejecutar(self):
        while self.corriendo:
            dt = self.clock.tick(FPS) / 1000.0

            self.manejar_eventos()
            self.actualizar(dt)
            self.dibujar()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    os.environ['SDL_VIDEODRIVER'] = 'windows'
    sim = Simulacion()
    sim.ejecutar()