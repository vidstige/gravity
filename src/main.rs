#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::f32::consts::TAU;
use rand::prelude::*;

use eframe::{egui::{self, Ui, Sense, Id}, epaint::{Color32, Pos2, Vec2, Stroke, Rect}, emath::NumExt};
use gravity::Simulation;
use gravity;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Gravity App",
        options,
        Box::new(|_| { Box::<GravityApp>::default() }),
    )
}

#[derive(PartialEq)]
enum Mode {
    Add,
    Remove,
}

struct GravityApp {
    rng: rand::rngs::ThreadRng,
    simulation: Simulation,
    play: bool,
    mode: Mode,
    radius: f32,
    orbital_velocity: bool,
}

impl Default for GravityApp {
    fn default() -> Self {
        Self {
            rng: rand::thread_rng(),
            simulation: Simulation::new(),
            play: false,
            mode: Mode::Add,
            radius: 64.0,
            orbital_velocity: true,
        }
    }
}

fn draw(ui: &mut Ui, simulation: &Simulation, spacing: Vec2) {
    let size = ui.available_size();
    
    // draw grid
    let color = Color32::from_rgb(00, 0x8b, 0x8b);
    let stroke = Stroke { width: 1.0, color};
    let grid_size = size / spacing;
    let (grid_width, grid_height) = (grid_size.x as usize, grid_size.y as usize);
    let painter = ui.painter();
    let d10 = Vec2::new(spacing.x, 0.0);
    let d01 = Vec2::new(0.0, spacing.y);
    for y in 0..grid_height + 1 {
        for x in 0..grid_width + 1 {
            let p = Pos2::new(x as f32 * spacing.x, y as f32 * spacing.y);
            painter.line_segment([p, p + d10], stroke);
            painter.line_segment([p, p + d01], stroke);
        }
    }

    // draw masses
    let color = Color32::from_rgba_premultiplied(00, 0x8b, 0x8b, 0x40);
    for position in simulation.state.positions.iter() {
        painter.circle_filled(Pos2::new(position.x, position.y), 3.0, color);
    }
}

fn random_point_in_circle(rng: &mut ThreadRng, radius: f32) -> Vec2 {
    let r = radius * rng.gen::<f32>().sqrt();
    let theta = rng.gen::<f32>() * TAU;
    Vec2::new(r * theta.cos(), r * theta.sin())
}

fn convert(v: Vec2) -> gravity::Vec2<f32> {
    gravity::Vec2 { x: v.x, y: v.y }
}

impl eframe::App for GravityApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let spacing = Vec2::new(32.0, 32.0);
        egui::SidePanel::right("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut self.play, "play");
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Add, "Add");
                ui.selectable_value(&mut self.mode, Mode::Remove, "Remove");
            });
            if self.mode == Mode::Add {
                ui.checkbox(&mut self.orbital_velocity, "orbital velocity");
            }
        });
        let id = Id::new("gravity_view");
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.play {
                // take a time step
                let dt = ui.input(|i| i.unstable_dt).at_most(1.0 / 30.0);
                gravity::step(&mut self.simulation, dt);
    
                // request a new timestep
                ui.ctx().request_repaint();
                //self.time += ;
            };

            draw(ui, &self.simulation, spacing);

            let rect = Rect::from_min_size(Pos2::ZERO, ui.available_size());
            
            ui.input(|i| self.radius = (self.radius + 0.1 * i.scroll_delta.y).clamp(0.0, 1024.0));

            let response = ui.interact(rect, id, Sense::hover());
            if let Some(hover_pos) = response.hover_pos() {
                let stroke = Stroke { width: 1.0, color: Color32::WHITE};
                ui.painter().circle_stroke(hover_pos, self.radius, stroke);
            }
            
            let response = ui.interact(rect, id, Sense::click_and_drag());
            if response.dragged() {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    match self.mode {
                        Mode::Add => {
                            let position = random_point_in_circle(&mut self.rng, self.radius) + pointer_pos.to_vec2();
                            let velocity = gravity::Vec2::zero();
                            self.simulation.add(&convert(position), &velocity, 1.0);
                        },
                        Mode::Remove => {
                            let center = convert(pointer_pos.to_vec2());
                            let r2 = self.radius * self.radius;
                            let mut indices = self.simulation.state.positions
                                .iter()
                                .enumerate()
                                .filter(|(_, &p)| p.sub(&center).norm2() < r2)
                                .map(|(index, _)| index)
                                .collect::<Vec<_>>();
                            indices.reverse();
                            for index in indices {
                                self.simulation.remove(index);
                            }
                        }                            
                    }
                }
            }
        });
    }
}