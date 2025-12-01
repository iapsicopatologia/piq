import 'package:brasil_fields/brasil_fields.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_modular/flutter_modular.dart';
import '/src/modelView/options_style/muro_colorido.dart';
import '/src/modelView/options_style/display_frame.dart';
import '/src/modelView/options_style/multi_selection_list.dart';

import '../../modelView/options_style/arvore_circulos.dart';
import '../../modelView/options_style/dots_line.dart';
import '../../modelView/options_style/find_images.dart';
import '../../modelView/options_style/five_errors.dart';
import '../../modelView/options_style/send_email.dart';
import '../../modelView/options_style/single_selection_list.dart';
import '../../modelView/options_style/text_form_list.dart';
import '../../modelView/options_style/yes_no.dart';
import 'telas_controller.dart';

void addListenerSimples(
  TelasController controller,
  GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
) {
  state.currentState!.didChange(controller.answerAux.value);
}

void addListenerComposto(
  TelasController controller,
  GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
  int i1,
  int i2,
) {
  if (controller.answerAux.value[i1].value == 'other') {
    if (controller.answerAux.value[i2].value == 'Sucess') {
      controller.answerAux.value[i2].value = '';
    }
  } else {
    if (controller.answerAux.value[i2].value == '') {
      controller.answerAux.value[i2].value = 'Sucess';
    }
  }
  state.currentState!.didChange(controller.answerAux.value);
}

Map<int, Map<String, dynamic>> telas = {
  1: {
    'hasProx': true,
    'header': "Psychopathological Interactive Questionnaire (PIQ)",
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const Text("""
FREE AND INFORMED CONSENT FORM

You are being invited to participate in the research titled “Systematization of tools for Psychopathological Assessment using Artificial Intelligence techniques”, under the responsibility of researchers Prof. Dr. Keiji Yamanaka and Ms. Résia da Silva Morais (from the School of Electrical Engineering at the Federal University of Uberlândia). This research aims to confirm the possibility of using Artificial Intelligence techniques to evaluate mental health structures called Psychic Mental Functions (attention, consciousness, orientation, time and space experience, sensory perception, memory, affectivity, will and psychomotricity, thought, reality judgment, language, intelligence, and social cognition), essential for high-quality psychopathological assessment to assist mental health professionals.

The Free and Informed Consent Form is being obtained by the researchers Dr. Keiji Yamanaka and Ms. Résia da Silva Morais before any experimentation related to the study. According to item IV of Resolution CNS 466/12 or Chapter III of Resolution 510/2016, the participant will have all the time needed to decide whether to participate in the research. If you have any questions regarding the study, you may contact the researchers via the email duvidaspsicopatologiacomia@gmail.com to discuss any information.

After understanding all information contained in this Consent Form and electronically selecting the option “I Agree”, screens containing instructions for the tasks you will answer will be displayed. Make sure you are in a quiet environment, free from distractions. Some screens will play sounds, so it is essential to use headphones or turn on your device’s speakers. The evaluation of the Psychic Functions will consist of 45 tests presented across 66 screens, and it will take approximately 25 minutes. In some questions, you will be instructed to listen to specific sounds before responding. Each screen of the interactive questionnaire will allow the identification of the relationship between your answers and possible signs of alterations in mental functions.

Your answers will be collected and recorded through Google Forms and the TensorFlow system, which will group the data and enable the analysis of the psychopathological assessment tools. The responsible researcher will follow the guidelines of Resolutions No. 466/2012, Chapter XI, Item XI.2: f and No. 510/2016, Chapter VI, Article 28: IV, and will store the research data in a physical or digital archive for a minimum of 5 (five) years after the conclusion of the research. After data collection is completed, the researcher will download the data and delete all records from any virtual platform, shared environment, or cloud. You will not be identified at any point. The study results may be published, and even then, your identity will be preserved.

There will be no cost or financial compensation for participating in the research. You will not receive payment for participating. All information obtained through your participation will be used exclusively for this research and will remain under the responsibility of the lead researcher. If any damage results from your participation, you may request compensation through legal means (Civil Code, Law 10.406/2002, Articles 927 to 954; and CNS Resolution No. 510/2016, Article 19).

The risks include possible tiredness during the tasks and some discomfort associated with the tests. To minimize these risks, the data collection instrument is answered virtually and anonymously. The researchers commit to maintaining confidentiality and data protection to preserve your identity. If you experience emotional discomfort during the test, feel free to contact the researchers at duvidaspsicopatologiacomia@gmail.com and you will receive support from the assistant researcher (a Clinical Psychologist). If necessary, you may receive brief psychotherapeutic guidance and be referred to public psychological services near your residence. The benefits include expanding your understanding of the importance of Artificial Intelligence as a supportive tool in psychological assessment, which may assist mental health professionals in clinical interventions. Research data will be stored physically and digitally for at least 5 (five) years after the end of the study, according to Chapter VI of CNS Resolution 510, April 7, 2016.

You are free to discontinue your participation at any time without any penalty or coercion. Until the study results are published, you may also request the removal of your data from the research. A copy of this Free and Informed Consent Form will remain with you, and you may save it on your device. For questions or complaints regarding the study, you may contact: Dr. Keiji Yamanaka, School of Electrical Engineering, Federal University of Uberlândia. Av. João Naves de Ávila, 2121, Santa Mônica Campus - Federal University of Uberlândia, ZIP 38400-902 - Uberlândia, MG - Brazil - Phone: +55 (34) 3239-4706, or the Psychologist Ms. Résia Silva de Morais at the same address, email: duvidaspsicopatologiacomia@gmail.com. To learn more about research participant rights, access the guidelines at: https://conselho.saude.gov.br/images/comissoes/conep/documentos/Cartilha_Direitos_Eticos_2020.pdf.

You may also contact the Human Research Ethics Committee (CEP) at the Federal University of Uberlândia, located at Av. João Naves de Ávila, 2121, Block A, Room 224, Santa Mônica Campus – Uberlândia/MG, ZIP 38408-100; Phone: +55 (34) 3239-4131; Email: cep@propp.ufu.br. The CEP is an independent committee created to defend the rights, integrity, and dignity of research participants and to ensure that research follows ethical standards established by the National Health Council.

By selecting the option “I Agree” below, you declare that you voluntarily agree to participate in the research after being properly informed by the researchers; that you understand the study; that you may request clarifications at any time via email (duvidaspsicopatologiacomia@gmail.com); and that you may withdraw at any time during or after your participation. You authorize the use of the data collected in this study while maintaining confidentiality of your identity.
""", textAlign: TextAlign.justify),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(() {
                if (controller.answerAux.value[0].value == 'I Agree') {
                  state.currentState!.didChange(controller.answerAux.value);
                } else {
                  state.currentState!.didChange([]);
                }
              }),
            hasPrefiroNaoDizer: false,
            options: const ['I Agree', 'I Do Not Agree'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Text("""
Please save this document for your records. If you wish to receive a copy of this consent form via email, please fill in your email below and click the send button:
""", textAlign: TextAlign.justify),
          const SizedBox(height: 10.0),
          SendEmail(answer: controller.emailAux),
          const Divider(),
          const SizedBox(height: 10.0),
          const Text("""
  RESEARCHERS’ DECLARATION

We declare that we have appropriately and voluntarily obtained the Free and Informed Consent of this participant for this study. We also declare that we commit to complying with all terms described herein.
""", textAlign: TextAlign.justify),
          const SizedBox(height: 10.0),
          Image.asset(
            'assets/assinatura_keiji.png',
            alignment: Alignment.bottomCenter,
          ),
          const SizedBox(height: 10.0),
          Image.asset(
            'assets/assinatura_resia.png',
            alignment: Alignment.bottomCenter,
          ),
        ],
  },

  // ----------------------------------------------------------------------
  2: {
    'hasProx': true,
    'header': 'Sociodemographic Questionnaire',
    'answerLenght': 19,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TextFormList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            labelText: 'What time is it right now?',
            validator: (value) {
              if (value == null) {
                return 'Invalid time!';
              } else if ((value.isEmpty) || (value.length != 5)) {
                return 'Invalid time!';
              }
              return null;
            },
            icon: Icons.lock_clock,
            keyboardType: TextInputType.number,
            inputFormatters: [
              FilteringTextInputFormatter.digitsOnly,
              HoraInputFormatter(),
            ],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
          TextFormList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            labelText: 'Today’s date?',
            validator: (value) {
              if (value == null) {
                return 'Incorrect date!';
              } else if ((value.isEmpty) || (value.length != 10)) {
                return 'Incorrect date!';
              }
              return null;
            },
            icon: Icons.date_range,
            keyboardType: TextInputType.number,
            inputFormatters: [
              FilteringTextInputFormatter.digitsOnly,
              DataInputFormatter(),
            ],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
          TextFormList(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            labelText: 'How old are you?',
            icon: Icons.cake,
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            validator: (String? value) {
              if ((value == null) ||
                  (value.isEmpty) ||
                  (int.parse(value) <= 0) ||
                  (int.parse(value) >= 130)) {
                return 'Invalid age! Please correct.';
              }
              return null;
            },
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[3]
              ..addListener(() => addListenerComposto(controller, state, 3, 4)),
            title: 'Gender *',
            icon: Icons.transgender,
            hasPrefiroNaoDizer: true,
            options: const ["Female", "Male"],
            optionsColumnsSize: 2,
            otherItem: TextFormList(
              answer: controller.answerAux.value[4]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              labelText: "What is your gender?",
              keyboardType: TextInputType.name,
              inputFormatters: [
                FilteringTextInputFormatter.singleLineFormatter,
              ],
              validator: (String? value) {
                if (value == null) {
                  return 'Invalid description! Please correct.';
                } else if ((value.isEmpty) || (value.length < 3)) {
                  return 'Invalid description! Please correct.';
                }
                return null;
              },
            ),
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[5]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'What sex was assigned to you at birth?',
            icon: Icons.wc,
            hasPrefiroNaoDizer: false,
            options: const ["Female", "Male"],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[6]
              ..addListener(() => addListenerComposto(controller, state, 6, 7)),
            title: "Indicate your Race or Ethnicity:",
            icon: Icons.person,
            hasPrefiroNaoDizer: true,
            options: const ["Black", "White", "Brown", "Yellow", "Indigenous"],
            optionsColumnsSize: 2,
            otherItem: TextFormList(
              answer: controller.answerAux.value[7]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              keyboardType: TextInputType.name,
              labelText: "What is your Race or Ethnicity?",
              inputFormatters: [
                FilteringTextInputFormatter.singleLineFormatter,
              ],
              validator: (value) {
                if (value == null) {
                  return 'Invalid description! Please correct.';
                } else if ((value.isEmpty) || (value.length < 3)) {
                  return 'Invalid description! Please correct.';
                }
                return null;
              },
            ),
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[8]
              ..addListener(() => addListenerComposto(controller, state, 8, 9)),
            title: "Within your family, are you an only child?",
            icon: Icons.diversity_3,
            hasPrefiroNaoDizer: false,
            options: const ["Yes"],
            optionsColumnsSize: 2,
            otherLabel: "No",
            otherItem: TextFormList(
              answer: controller.answerAux.value[9]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              labelText: "How many siblings do you have?",
              keyboardType: TextInputType.number,
              inputFormatters: [FilteringTextInputFormatter.digitsOnly],
              validator: (value) {
                if (value == null) {
                  return 'Invalid quantity! Please correct.';
                } else if ((value.isEmpty)) {
                  return 'Invalid quantity! Please correct.';
                }
                return null;
              },
            ),
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[10]
              ..addListener(
                () => addListenerComposto(controller, state, 10, 11),
              ),
            title: "What is your marital status?",
            icon: Icons.diversity_2,
            hasPrefiroNaoDizer: false,
            optionsColumnsSize: 2,
            options: const [
              "Single",
              "Married",
              "Widowed",
              "Divorced",
              "Common-law union",
            ],
            otherItem: TextFormList(
              answer: controller.answerAux.value[11]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              labelText: "Specify your marital status:",
              keyboardType: TextInputType.name,
              inputFormatters: [
                FilteringTextInputFormatter.singleLineFormatter,
              ],
              validator: (value) {
                if (value == null) {
                  return 'Invalid description! Please correct.';
                } else if ((value.isEmpty) || (value.length < 3)) {
                  return 'Invalid description! Please correct.';
                }
                return null;
              },
            ),
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[12]
              ..addListener(
                () => addListenerComposto(controller, state, 12, 13),
              ),
            title: "Do you have children?",
            icon: Icons.group_add,
            hasPrefiroNaoDizer: false,
            optionsColumnsSize: 2,
            options: const ["No"],
            otherLabel: "Yes",
            otherItem: TextFormList(
              answer: controller.answerAux.value[13]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              labelText: "How many children do you have?",
              keyboardType: TextInputType.number,
              inputFormatters: [FilteringTextInputFormatter.digitsOnly],
              validator: (value) {
                if (value == null) {
                  return 'Invalid quantity! Please correct.';
                } else if ((value.isEmpty)) {
                  return 'Invalid quantity! Please correct.';
                }
                return null;
              },
            ),
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[14]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: "Do you have children under 6 years old?",
            icon: Icons.child_friendly,
            hasPrefiroNaoDizer: false,
            optionsColumnsSize: 2,
            options: const ["No", "Yes"],
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[15]
              ..addListener(
                () => addListenerComposto(controller, state, 15, 16),
              ),
            title: "Religion *",
            icon: Icons.church,
            hasPrefiroNaoDizer: false,
            optionsColumnsSize: 2,
            options: const ["No religion"],
            otherLabel: "I have a religion",
            otherItem: TextFormList(
              answer: controller.answerAux.value[16]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              labelText: "Which religion?",
              keyboardType: TextInputType.name,
              inputFormatters: [
                FilteringTextInputFormatter.singleLineFormatter,
              ],
              validator: (value) {
                if (value == null) {
                  return 'Invalid religion! Please correct.';
                } else if ((value.isEmpty)) {
                  return 'Invalid religion! Please correct.';
                }
                return null;
              },
            ),
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[17]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: "Education *",
            icon: Icons.school,
            hasPrefiroNaoDizer: false,
            options: const [
              "No schooling",
              "Elementary School (1st level) incomplete",
              "Elementary School (1st level) complete",
              "High School (2nd level) incomplete",
              "High School (2nd level) complete",
              "Undergraduate incomplete",
              "Undergraduate complete",
              "Master’s degree",
              "Doctorate",
            ],
          ),

          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[18]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: "Your household’s monthly family income",
            icon: Icons.attach_money,
            hasPrefiroNaoDizer: false,
            options: const [
              "Up to 1 minimum wage",
              "More than 1 to 2 minimum wages",
              "More than 2 to 3 minimum wages",
              "More than 3 to 5 minimum wages",
              "More than 5 to 8 minimum wages",
              "More than 8 to 12 minimum wages",
              "More than 12 to 20 minimum wages",
              "More than 20 minimum wages",
            ],
          ),
        ],
  },
  3: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 3,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(() => addListenerComposto(controller, state, 0, 1)),
            title:
                'Do you have any mental health diagnosis confirmed by a licensed health professional?',
            optionsColumnsSize: 2,
            hasPrefiroNaoDizer: false,
            options: const ["No"],
            otherLabel: "Yes",
            otherItem: SingleSelectionList(
              answer: controller.answerAux.value[1]
                ..addListener(
                  () => addListenerComposto(controller, state, 1, 2),
                ),
              title: "\nIf yes, select the corresponding diagnosis.",
              icon: Icons.admin_panel_settings,
              hasPrefiroNaoDizer: false,
              options: const [
                "Autism Spectrum Disorder",
                "Depressive Disorders",
                "Cyclothymic Disorder",
                "Anxiety Disorders",
                "Post-traumatic Stress Disorder",
                "Eating Disorders",
                "Bipolar Disorder",
                "Obsessive-Compulsive Disorder",
                "Attention-Deficit/Hyperactivity Disorder",
                "Borderline Personality Disorder",
                "Schizophrenia Spectrum and Other Psychotic Disorders",
              ],
              otherLabel: "Other disorder",
              otherItem: TextFormList(
                answer: controller.answerAux.value[2]
                  ..addListener(
                    () => state.currentState!.didChange(
                      controller.answerAux.value,
                    ),
                  ),
                labelText: 'Type the name of this other disorder',
                keyboardType: TextInputType.name,
                inputFormatters: [
                  FilteringTextInputFormatter.singleLineFormatter,
                ],
                validator: (value) {
                  if (value == null) {
                    return 'Please type the name of this disorder';
                  } else if (value.length < 4) {
                    return 'Please type the name of this disorder';
                  }
                  return null;
                },
              ),
            ),
          ),
        ],
  },

  4: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            'From this point on, screens will be presented with the instructions for the tasks you will answer.\r\n\nMake sure you are in a quiet environment, free from distractions.\f\n\nIn some screens, sounds will be played; therefore, it is essential to use headphones or turn on your device speakers.',
        bodyHasFrame: false,
      ),
    ],
  },

  5: {
    'hasProx': true,
    'header': 'Information',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body: 'Look carefully at the figure presented on the next screen.',
        bodyHasFrame: false,
      ),
    ],
  },

  6: {
    'hasProx': false,
    'header': 'Observe',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/arvore_free.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },

  7: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'What did you see on the previous screen?',
            icon: Icons.question_answer,
            hasPrefiroNaoDizer: false,
            options: const [
              "Jesus Christ",
              "Heart",
              "Dragon breathing fire",
              "Tree",
              "I didn’t see anything",
              "Something else",
            ],
            optionsColumnsSize: 2,
          ),
        ],
  },

  8: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body: 'Look carefully at the figure presented on the next screen.',
        bodyHasFrame: false,
      ),
    ],
  },

  9: {
    'hasProx': false,
    'header': 'Observe',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [const DisplayFrame(body: '', bodyHasFrame: true)],
  },

  10: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'What did you see on the previous screen?',
            icon: Icons.timelapse,
            hasPrefiroNaoDizer: false,
            options: const [
              "Jesus Christ",
              "Heart",
              "Dragon breathing fire",
              "Tree",
              "I didn’t see anything",
              "Something else",
            ],
            optionsColumnsSize: 2,
          ),
        ],
  },

  11: {
    'hasProx': true,
    'header': 'Information',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            "On the next screens, some number sequences will be presented. After viewing them, you must select the answer that corresponds to the correct sequence.",
        bodyHasFrame: false,
      ),
    ],
  },

  12: {
    'hasProx': false,
    'header': '',
    'delay': 3,
    'answerLenght': 1,
    'itens': (_, __) => [const DisplayFrame(body: '2 - 7', bodyHasFrame: true)],
  },

  13: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'What was the correct sequence of numbers shown on the previous screen?',
            icon: Icons.timeline,
            hasPrefiroNaoDizer: false,
            options: const [
              "1 - 5",
              "4 - 7",
              "2 - 7",
              "2 - 8",
              "9 - 4",
              "7 - 2",
            ],
            optionsColumnsSize: 3,
          ),
        ],
  },

  14: {
    'hasProx': false,
    'header': '',
    'delay': 3,
    'answerLenght': 1,
    'itens': (_, __) => [
      const DisplayFrame(body: '5 - 6 - 4', bodyHasFrame: true),
    ],
  },

  15: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'What was the correct sequence of numbers shown on the previous screen?',
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              "5 - 7 - 1",
              "1 - 3 - 4",
              "5 - 6 - 3",
              "4 - 6 - 5",
              "5 - 4 - 6",
              "5 - 6 - 4",
            ],
            optionsColumnsSize: 3,
          ),
        ],
  },

  16: {
    'hasProx': false,
    'header': '',
    'delay': 3,
    'answerLenght': 1,
    'itens': (_, __) => [
      const DisplayFrame(body: '6 - 4 - 3 - 9', bodyHasFrame: true),
    ],
  },

  17: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'What was the correct sequence of numbers shown on the previous screen?',
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              "6 - 4 - 9 - 3",
              "4 - 8 - 9 - 1",
              "6 - 4 - 3 - 9",
              "5 - 4 - 3 - 8",
              "1 - 5 - 2 - 9",
              "6 - 4 - 3 - 7",
            ],
            optionsColumnsSize: 2,
          ),
        ],
  },

  18: {
    'hasProx': false,
    'header': '',
    'delay': 3,
    'answerLenght': 1,
    'itens': (_, __) => [
      const DisplayFrame(body: '4 - 2 - 7 - 3 - 1', bodyHasFrame: true),
    ],
  },

  19: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'What was the correct sequence of numbers shown on the previous screen?',
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              "2 - 1 - 4 - 7 - 9",
              "4 - 3 - 9 - 8",
              "4 - 2 - 6 - 3 - 1",
              "7 - 5 - 1 - 4 - 2 ",
              "4 - 2 - 7 - 3 - 1",
              "6 - 3 - 1 - 5 - 9",
            ],
            optionsColumnsSize: 2,
          ),
        ],
  },
  20: {
    'hasProx': false,
    'header': '',
    'delay': 3,
    'answerLenght': 1,
    'itens': (_, __) => [
      const DisplayFrame(body: '6 - 1 - 9 - 4 - 7 - 3', bodyHasFrame: true),
    ],
  },
  21: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'What was the correct sequence of numbers shown on the previous screen?',
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              "6 - 1 - 4 - 7 - 3 - 9",
              "2 - 1 - 8 - 3 - 9 - 5",
              "6 - 1 - 9 - 4 - 7 - 3",
              "6 - 4 - 5 - 8 - 3 - 7",
              "2 - 8 - 6 - 4 - 7 - 3",
              "6 - 1 - 9 - 4 - 5 - 2",
            ],
            optionsColumnsSize: 1,
          ),
        ],
  },
  22: {
    'hasProx': false,
    'header': '',
    'delay': 3,
    'answerLenght': 1,
    'itens': (_, __) => [
      const DisplayFrame(body: '5 - 9 - 1 - 7 - 4 - 2 - 8', bodyHasFrame: true),
    ],
  },
  23: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'What was the correct sequence of numbers shown on the previous screen?',
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              "5 - 8 - 1 - 7 - 4 - 3 - 8",
              "5 - 9 - 1 - 7 - 4 - 8",
              "8 - 9 - 0 - 7 - 3 - 1",
              "5 - 9 - 1 - 7 - 4 - 2 - 8",
              "5 - 2 - 3 - 7 - 4 - 9 - 8",
              "5 - 9 - 1 - 7 - 8 - 0 - 9",
            ],
            optionsColumnsSize: 1,
          ),
        ],
  },
  24: {
    'hasProx': true,
    'header': 'Information',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            "Please identify the expressions and intentions of the people by considering only the eye region. For each image, select the word that best describes the feelings, thoughts, or impressions the person seems to be expressing.",
        bodyHasFrame: false,
      ),
    ],
  },
  25: {
    'hasProx': true,
    'header':
        'Among the four alternatives in each image, select the word that best describes it.',
    'answerLenght': 14,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(body: 'assets/olho1.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              'Restless',
              'Thoughtful',
              'Irritated',
              'Suspicious',
            ],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho2.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Nervous', 'Depressed', 'Irritated', 'Amused'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho3.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Embarrassed', 'Amused', 'Interested', 'Depressed'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho4.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[3]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Arrogant', 'Determined', 'Terrified', 'Upset'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho5.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[4]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Kind', 'Determined', 'Friendly', 'Depressed'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho6.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[5]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Shy', 'Disturbed', 'Discouraged', 'Thoughtful'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho7.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[6]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              'Impatient',
              'Discouraged',
              'Seductive',
              'Relieved',
            ],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho8.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[7]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Grateful', 'Dreamy', 'Discouraged', 'Shocked'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho9.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[8]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              'Satisfied',
              'Worried',
              'Affectionate',
              'Indifferent',
            ],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho10.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[9]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Kind', 'Regretful', 'Angry', 'Friendly'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho11.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[10]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Uncomfortable', 'Bored', 'Confident', 'Impatient'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho12.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[11]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Regretful', 'Nervous', 'Amused', 'Embarrassed'],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho13.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[12]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const [
              'Friendly',
              'Entertained',
              'Suspicious',
              'Seductive',
            ],
            optionsColumnsSize: 2,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          const DisplayFrame(body: 'assets/olho14.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[13]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            hasPrefiroNaoDizer: false,
            options: const ['Worried', 'Cautious', 'Hostile', 'Amused'],
            optionsColumnsSize: 2,
          ),
        ],
  },
  26: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Have you felt intense fear of losing control or going crazy?',
              'To prevent weight gain, do you use laxatives, diuretics, or other medications; fasting, self-induced vomiting, or excessive exercise?',
              'Do you feel intense fear of gaining weight or getting fat, to the point of not eating?',
              'Do you have persistent ingestion of non-nutritive substances such as sweets and/or chocolate for a minimum period of one month?',
              'Do you get irritated easily, so that your mood changes quickly during the day?',
              'Have you had episodes of intense fear or panic that made you feel very unwell?',
            ],
          ),
        ],
  },
  27: {
    'hasProx': true,
    'header': 'Which of the images below completes the sequence?',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(body: 'assets/intel_1.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            options: const [
              'assets/intel_1a.png',
              'assets/intel_1b.png',
              'assets/intel_1c.png',
              'assets/intel_1d.png',
              'assets/intel_1e.png',
              'assets/intel_1f.png',
            ],
            optionsColumnsSize: 3,
            maxSizeAnswer: 1,
            mimSizeAnswer: 1,
          ),
        ],
  },
  28: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Have you recently pulled your own hair repeatedly, resulting in hair loss?',
              'Have you experienced a great loss of interest or pleasure in all or almost all activities most of the day; feeling sad, empty, or hopeless?',
              'Are you bothered by thoughts that come into your mind even when you do not want them, such as fear of being exposed to germs, diseases, or dirt, or the need for everything to be arranged in a certain way?',
              'Are you bothered by unwanted images in your mind, such as violent and horrible scenes, or content of a sexual nature?',
              'Do you feel that you are a special and unique person? Do you hope that one day people will recognize your value and the difference you make in their lives?',
              'Do you tend to see yourself as socially incapable, lacking personal appeal, or inferior to others?',
              'Do you have difficulty initiating projects or doing things alone (due to lack of self-confidence rather than lack of motivation or energy)?',
            ],
          ),
        ],
  },
  29: {
    'hasProx': true,
    'header': 'Complete the following sequences:',
    'answerLenght': 4,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TextFormList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.confirmation_num,
            title: '2, 4, 8, 16, ?',
            labelText: "Sequence *",
            optionsColumnsSize: 1,
            validator: (value) {
              if (value == null) {
                return 'Invalid value!';
              } else if (value.isEmpty) {
                return 'Invalid value!';
              }
              return null;
            },
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.confirmation_num,
            title: '1, 3, 9, ?',
            labelText: "Sequence *",
            optionsColumnsSize: 1,
            validator: (value) {
              if (value == null) {
                return 'Invalid value!';
              } else if (value.isEmpty) {
                return 'Invalid value!';
              }
              return null;
            },
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.confirmation_num,
            title: '3, 7, 11, 15, ? ',
            labelText: "Sequence *",
            optionsColumnsSize: 1,
            validator: (value) {
              if (value == null) {
                return 'Invalid value!';
              } else if (value.isEmpty) {
                return 'Invalid value!';
              }
              return null;
            },
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[3]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.confirmation_num,
            title: '32, 16, 8, ? ',
            labelText: "Sequence *",
            optionsColumnsSize: 1,
            validator: (value) {
              if (value == null) {
                return 'Invalid value!';
              } else if (value.isEmpty) {
                return 'Invalid value!';
              }
              return null;
            },
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
        ],
  },

  30: {
    //To insert characters in a String use: "\u{0x___}"
    'hasProx': true,
    'header': 'Observe the words below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                '\n1)\tMANY\n2)\tOCEAN\n3)\tFISH\n4)\tAND\n5)\tHAS\n6)\tTHE\n7)\tPLANTS\n\nNow form a meaningful sentence containing all these words. Select the correct order:',
            hasPrefiroNaoDizer: false,
            options: const [
              '1 - 4 - 6 - 2 - 5 - 3 - 7',
              '6 - 2 - 4 - 7 - 5 - 1 - 3',
              '5 - 1 - 3 - 4 - 6 - 7 - 2',
              '6 - 2 - 5 - 1 - 3 - 4 - 7',
              '7 - 4 - 3 - 5 - 1 - 2 - 6 ',
            ],
            optionsColumnsSize: 1,
          ),
        ],
  },
  31: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Do you feel something strange inside your body, such as movements or pushes?',
              'Do you feel as if someone touches, pinches, hits, or kisses your body?',
              'Do you feel a lack of control while eating; unable to stop or control the amount you eat?',
              'Do you feel rested and ready for the day even after only 3 hours of sleep?',
              'Do you frequently have difficulty staying focused during tasks or activities?',
              'Do you often answer questions hastily before they have been fully asked?',
            ],
          ),
        ],
  },
  32: {
    'hasProx': true,
    'header':
        'Fill in the field below with the city and state where you are now.',
    'answerLenght': 2,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TextFormList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.location_city,
            labelText: 'CITY:',
            optionsColumnsSize: 1,
            validator: (value) {
              if (value == null) {
                return 'Invalid City!!';
              } else if (value.isEmpty) {
                return 'Invalid City!!';
              }
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),
          TextFormList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.location_history,
            labelText: 'STATE:',
            optionsColumnsSize: 1,
            validator: (value) {
              if (value == null) {
                return 'Invalid State!!';
              } else if (value.isEmpty) {
                return 'Invalid State!!';
              }
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
        ],
  },
  33: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'Select the current time period of the day:',
            icon: Icons.timer_outlined,
            hasPrefiroNaoDizer: false,
            options: const [
              'Morning: 6:00 to 11:59',
              'Afternoon: 12:00 to 17:59',
              'Evening: 18:00 to 23:59',
              'Dawn: 00:00 to 05:59',
            ],
            optionsColumnsSize: 1,
          ),
        ],
  },
  34: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            'Six (6) images will be displayed. Stay alert! At a certain point in the test, they will be hidden inside a picture and you will need to find them.',
        bodyHasFrame: false,
      ),
    ],
  },
  35: {
    'hasProx': false,
    'header': 'Attention!!',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/CORREDOR.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },
  36: {
    'hasProx': false,
    'header': 'Attention!!',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/PAPAI NOEL.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },
  37: {
    'hasProx': false,
    'header': 'Attention!!',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/circo.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },
  38: {
    'hasProx': false,
    'header': 'Attention!!',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/chuteira.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },
  39: {
    'hasProx': false,
    'header': 'Attention!!',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/NUMERO8.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },

  40: {
    'hasProx': false,
    'header': 'Attention!!',
    'answerLenght': 0,
    'itens': (TelasController controller, __) => [
      DisplayFrame(
        body: 'assets/reciclagem.png',
        bodyHasFrame: true,
        isLoading: () {
          Future.delayed(const Duration(seconds: 3)).then((value) {
            Modular.to.popAndPushNamed(
              '/',
              arguments: controller.idPage.value + 1,
            );
          });
        },
      ),
    ],
  },
  41: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Daily, are there moments when you feel as if the ground is shaking?',
              'During the day, do you often feel irritated and impatient?',
              'Daily, are there moments when you feel very happy, radiant?',
              'During the day, are there several moments when you feel sad and/or melancholic?',
              'Have you noticed excessive concern about one or more perceived defects in your physical appearance that are not observable or appear minor to others?',
              'Do you often fidget with your hands or feet, or squirm in your seat while sitting?',
              'Do you show difficulty agreeing with the ideas of others, demonstrating inflexibility and rigidity toward yourself and others?',
            ],
          ),
        ],
  },
  42: {
    'hasProx': true,
    'header': 'Find the objects!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(
            body:
                'Six (6) images were shown earlier in the test. Can you find them? Click on the images you remember. You are not required to find them all. Do your best!\n',
            bodyHasFrame: false,
          ),
          const SizedBox(height: 10.0),
          FindImages(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            imagem: 'assets/seisimagens.png',
          ),
        ],
  },
  43: {
    'hasProx': true,
    'header': 'Evaluate and answer!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(body: "assets/questao32.png", bodyHasFrame: true),
          const SizedBox(height: 10.0),
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                "\nIf you had the opportunity, for a few minutes, to gain knowledge about someone or something through an opening in time and space, which of the following actions would you choose? Pick two or three options that best match you.\n",
            options: const [
              'Meet with a deceased loved one.',
              'See a person naked.',
              'Know how to perform daily tasks without being harmed by incessant thoughts that limit me.',
              'Go back to my childhood and start everything over.',
              'Go back to my teenage years and start everything over.',
              'Know whether my boyfriend/girlfriend or spouse is cheating on me.',
              'Know what my family’s future will be like and help them.',
              'Disappear into time and space by entering the rift.',
              'Know if I will be alive five years from now.',
              'Know who my soulmate is.',
              'See my professional future.',
              'Know how to stop thinking bizarre things.',
              'Witness future technological advancements.',
              'Witness a major historical event, such as the building of the Egyptian pyramids, the Great Wall of China, the age of dinosaurs, or inventions by Albert Einstein or Leonardo da Vinci.',
              'Know if I will become rich.',
              'Know who is following me on the street.',
              'Know how to sleep through the whole night.',
              'Know how to make my spouse and/or child make choices based on my values and principles.',
              'Know how to make my existential pain and suffering disappear.',
            ],
            optionsColumnsSize: 1,
            mimSizeAnswer: 2,
            maxSizeAnswer: 3,
          ),
        ],
  },
  44: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Do you perceive yourself as impulsive and reckless; e.g., excessive spending, sex with strangers, substance abuse, reckless driving, or repetitive self-harm?',
              'Do you have difficulty finishing what you start?',
              'When you tell a story about something that happened, do you often exaggerate or dramatize details?',
              'Have you witnessed or still been repeatedly or intensely exposed to aversive details of a traumatic event?',
              'Do you have recurrent outbursts of anger, expressed verbally (such as insults) or physically (such as aggression)?',
              'Are you experiencing insomnia almost every day?',
            ],
          ),
        ],
  },
  45: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body: 'Pay attention to the sound that will play on the next screen.',
        bodyHasFrame: false,
      ),
    ],
  },
  46: {
    'hasProx': false,
    'header': 'Press play to listen!!',
    'answerLenght': 0,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          DisplayFrame(
            id: controller.idPage.value,
            body: "assets/audios/aguacorrente-edited_v2.mp3",
            bodyHasFrame: true,
            playMusic: controller.playMusic,
          ),
        ],
  },
  47: {
    'hasProx': true,
    'header': 'Answer !!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'Which option matches the sound you heard?',
            hasPrefiroNaoDizer: false,
            options: const [
              "Birds",
              "Water sound",
              "Vacuum cleaner",
              "Child crying",
              "Phone ringing",
              "No sound",
            ],
            optionsColumnsSize: 2,
          ),
        ],
  },
  48: {
    'hasProx': true,
    'header': 'Observe and answer',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(body: "assets/Ebbinghaus.png", bodyHasFrame: true),
          const SizedBox(height: 10.0),
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'Select the option that corresponds to what you see in the image above.',
            icon: Icons.more_time,
            options: const [
              'Line A is longer than Line B',
              'Line A is shorter than Line B',
              'Lines A and B are the same size',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  49: {
    'hasProx': true,
    'header': 'Draw !!',
    'answerLenght': 3,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(() {
                if (controller.answerAux.value[0].value ==
                    'I do not wish to do it') {
                  controller.answerAux.value[1].value = 'Success';
                  controller.answerAux.value[2].value = 'Success';
                  state.currentState!.didChange(controller.answerAux.value);
                } else {
                  addListenerComposto(controller, state, 0, 1);
                  controller.answerAux.value[2].value = '';
                }
              }),
            title:
                'Several points will be displayed, and by clicking on them you can create images as the points connect with straight lines. Choose one of the suggestions, then begin drawing. The available time is flexible — you may return or erase as needed.',
            options: const [
              'Airplane',
              'Butterfly',
              'House',
              'Star',
              'Square',
              'I do not wish to do it',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
            otherItem: TextFormList(
              answer: controller.answerAux.value[1]
                ..addListener(
                  () =>
                      state.currentState!.didChange(controller.answerAux.value),
                ),
              keyboardType: TextInputType.name,
              labelText: "What do you wish to draw?",
              inputFormatters: [
                FilteringTextInputFormatter.singleLineFormatter,
              ],
              validator: (value) {
                if (value == null) {
                  return 'Invalid description!! Please correct it';
                } else if ((value.isEmpty) || (value.length < 3)) {
                  return 'Invalid description!! Please correct it';
                }
                return null;
              },
            ),
          ),
          const SizedBox(height: 10.0),
          DotsLine(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
          ),
        ],
  },
  50: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Have you been feeling more fatigue or low energy nearly every day?',
              'Have you experienced a traumatic event?',
              'Have you been hearing ringing in your ear?',
              'Do you have difficulty discarding used or worthless objects, even when they have no sentimental value? Do you keep many items, papers, receipts, thinking they might be useful someday?',
              'Do you exhibit repetitive behaviors (e.g., washing hands, organizing, checking) or mental acts (e.g., praying, counting, repeating words silently)?',
              'Do you feel bothered by impulses such as hurting someone you love, even though you do not want to?',
              'Have you seen anything unusual such as figures, shadows, fire, ghosts, demons, strange people, or similar in your daily life?',
            ],
          ),
        ],
  },
  51: {
    'hasProx': true,
    'leading': {
      'selectedIcon': Icons.comment_sharp,
      'deselectedIcon': Icons.comments_disabled,
    },
    'header': 'Find the 5 differences game',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(
            body:
                'The button in the app bar above allows switching between two images: one "real" and the other "altered." Between them are 5 differences you must identify by clicking the corresponding locations on the altered image.',
            bodyHasFrame: false,
          ),
          const SizedBox(height: 10.0),
          FiveErrors(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            imagemFull: 'assets/five_errors1.jpg',
            imagemClean: 'assets/five_errors2.jpg',
            isImagemFull: controller.isImagemFull,
          ),
        ],
  },
  52: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'In recent months, have you heard voices from unknown people?',
              'Lately, have you heard your own thoughts being spoken out loud?',
              'Is someone trying to poison you?',
              'Do you have difficulty relaxing? Are you always busy?',
              'Have you been experiencing sensations of shortness of breath or suffocation?',
              'Are you experiencing prolonged and persistent grief for over 12 months, marked by intense longing, worry, and apathy about the future?',
              'Are you constantly afraid of being fired?',
            ],
          ),
        ],
  },
  53: {
    'hasProx': true,
    'header': 'Evaluate and answer!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(body: 'assets/intel_2.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                "Which of the images below completes the following sequence?",
            options: const [
              'assets/intel_2a.png',
              'assets/intel_2b.png',
              'assets/intel_2c.png',
              'assets/intel_2d.png',
              'assets/intel_2e.png',
              'assets/intel_2f.png',
            ],
            optionsColumnsSize: 3,
            mimSizeAnswer: 1,
            maxSizeAnswer: 1,
          ),
        ],
  },

  54: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Do you live in fear of disappointing people?',
              'Have you been getting irritated more easily than before?',
              'In recent months, have you found it hard to stop worrying?',
              'Do you think about many things at the same time?',
              'Do you have difficulty concentrating?',
              'Do you avoid professional or academic activities that require interpersonal contact because you fear criticism, disapproval, or rejection?',
              'Do you believe you have no weaknesses and never avoid environments, because you can achieve anything you want?',
            ],
          ),
        ],
  },
  55: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body: "On the next screen, some words will be played. Pay attention.",
        bodyHasFrame: false,
      ),
    ],
  },
  56: {
    'hasProx': false,
    'header': 'Press play to listen!!',
    'answerLenght': 0,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          DisplayFrame(
            id: controller.idPage.value,
            body: 'assets/audios/quatro_palavras.mp3',
            bodyHasFrame: true,
            playMusic: controller.playMusic,
          ),
        ],
  },
  57: {
    'hasProx': true,
    'header': 'Evaluate and answer!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'Choose 2 to 3 images that best represent you or your personality.',
            options: const [
              'assets/questao45coquetel.png',
              'assets/questao45humburgue.png',
              'assets/questao45casa.png',
              'assets/questao45gato.png',
              'assets/questao45trabalhador.png',
              'assets/questao45carro.png',
              'assets/questao45cachorro.png',
              'assets/questao45passaro.png',
              'assets/questao45cerveja.png',
              'assets/questao45cocacola.png',
              'assets/questao45cafe.png',
              'assets/questao45bombons.png',
              'assets/questao45viajar.png',
              'assets/questao45livros.png',
              'assets/questao45leaonovo.png',
              'assets/questao45controlegame.png',
            ],
            optionsColumnsSize: 2,
            mimSizeAnswer: 2,
            maxSizeAnswer: 3,
          ),
        ],
  },
  58: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Do you believe you must do things perfectly, otherwise you will not be accepted?',
              'Do you believe people are judging your actions and behaviors even when they do not say so?',
              'Do you have negative thoughts such as thinking about dying?',
              'Do you feel inadequate, restless, and talk excessively?',
              'Do you often postpone or avoid doing things until the last minute? Do you tend to procrastinate?',
              'Have you been reluctant to delegate tasks or work with others because you feel you must control everything?',
              'Do you notice that you make great efforts to avoid being abandoned?',
            ],
          ),
        ],
  },
  59: {
    'hasProx': true,
    'header': 'Answer according to the sound previously heard!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'Which of the options matches the sound you heard?\n',
            icon: Icons.more_time,
            options: const [
              "Rua - Madeira - Paz - Pastel",
              "Lua - Cadeira - Raiz - Chapéu",
              "Rua - Cadeira - Paz - Chapéu",
              "Lua - Cadeira - Paz - Pastel",
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  60: {
    'hasProx': true,
    'header':
        'Choose the expression that best matches the basic emotion described.',
    'answerLenght': 6,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(body: 'assets/questao48.png', bodyHasFrame: true),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          // Disgust
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'a) Disgust:',
            icon: Icons.more_time,
            options: const ["1", "2", "3", "4", "5", "6"],
            optionsColumnsSize: 6,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          // Sadness
          SingleSelectionList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'b) Sadness:',
            icon: Icons.more_time,
            options: const ["1", "2", "3", "4", "5", "6"],
            optionsColumnsSize: 6,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          // Fear
          SingleSelectionList(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'c) Fear:',
            icon: Icons.more_time,
            options: const ["1", "2", "3", "4", "5", "6"],
            optionsColumnsSize: 6,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          // Anger
          SingleSelectionList(
            answer: controller.answerAux.value[3]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'd) Anger:',
            icon: Icons.more_time,
            options: const ["1", "2", "3", "4", "5", "6"],
            optionsColumnsSize: 6,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          // Joy
          SingleSelectionList(
            answer: controller.answerAux.value[4]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'e) Joy:',
            icon: Icons.more_time,
            options: const ["1", "2", "3", "4", "5", "6"],
            optionsColumnsSize: 6,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          // Surprise
          SingleSelectionList(
            answer: controller.answerAux.value[5]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: 'f) Surprise:',
            icon: Icons.more_time,
            options: const ["1", "2", "3", "4", "5", "6"],
            optionsColumnsSize: 6,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  61: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            "Over the next 4 screens, several expressions representing different feelings will be shown. On each screen, select at least 3 and at most 4 expressions that best represent what you have felt in the past few months.\n\nWhen you're ready, click next...",
        bodyHasFrame: false,
      ),
    ],
  },
  62: {
    'hasProx': true,
    'header': 'Select!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'Select at least 2 and at most 4 expressions that best represent how you have felt in the past few months.',
            options: const [
              'assets/emoji_sempre_atrasado.png',
              'assets/emoji_poderoso.png',
              'assets/emoji_otimista.png',
              'assets/emoji_em_panico.png',
              'assets/emoji_indeciso.png',
              'assets/emoji_triste.png',
              'assets/emoji_entediado.png',
              'assets/emoji_pessimista.png',
              'assets/emoji_reflexivo.png',
              'assets/emoji_confiante.png',
              'assets/emoji_forte.png',
              'assets/emoji_nojo.png',
            ],
            optionsColumnsSize: 3,
            mimSizeAnswer: 2,
            maxSizeAnswer: 4,
          ),
        ],
  },
  63: {
    'hasProx': true,
    'header': 'Select!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'Select at least 2 and at most 4 expressions that best represent how you have felt in the past few months.',
            options: const [
              'assets/emoji_esperancoso.png',
              'assets/emoji_pura_alegria.png',
              'assets/emoji_burro.png',
              'assets/emoji_surpreso.png',
              'assets/emoji_feliz.png',
              'assets/emoji_velho.png',
              'assets/emoji_frustrado.png',
              'assets/emoji_ansioso.png',
              'assets/emoji_emocionado.png',
              'assets/emoji_em_paz.png',
              'assets/emoji_inteligente.png',
              'assets/emoji_preocupado.png',
            ],
            optionsColumnsSize: 3,
            mimSizeAnswer: 2,
            maxSizeAnswer: 4,
          ),
        ],
  },
  64: {
    'hasProx': true,
    'header': 'Select!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'Select at least 2 and at most 4 expressions that best represent how you have felt in the past few months.',
            options: const [
              'assets/emoji_apaixonado.png',
              'assets/emoji_desesperado.png',
              'assets/emoji_envergonhado.png',
              'assets/emoji_abencoado.png',
              'assets/emoji_impulsivo.png',
              'assets/emoji_amado.png',
              'assets/emoji_confuso.png',
              'assets/emoji_sem_forcas.png',
              'assets/emoji_sonolento.png',
              'assets/emoji_gordo.png',
              'assets/emoji_sem_paciencia.png',
              'assets/emoji_com_ciumes.png',
            ],
            optionsColumnsSize: 3,
            mimSizeAnswer: 2,
            maxSizeAnswer: 4,
          ),
        ],
  },
  65: {
    'hasProx': true,
    'header': 'Select!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                'Select at least 2 and at most 4 expressions that best represent how you have felt in the past few months.',
            options: const [
              'assets/emoji_com_raiva.png',
              'assets/emoji_silenciado.png',
              'assets/emoji_desanimado.png',
              'assets/emoji_com_gratidao.png',
              'assets/emoji_fraude.png',
              'assets/emoji_agressivo.png',
              'assets/emoji_desequilibrado.png',
              'assets/emoji_comendo_muito.png',
              'assets/emoji_fumando_muito.png',
              'assets/emoji_jesus.png',
              'assets/emoji_bebendo_muito.png',
              'assets/emoji_animado.png',
            ],
            optionsColumnsSize: 3,
            mimSizeAnswer: 2,
            maxSizeAnswer: 4,
          ),
        ],
  },
  66: {
    'hasProx': true,
    'header': 'Evaluate and answer!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          MultiSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title:
                "Select the words you like the most and that also describe you. The amount is unlimited — choose as many words as you want, as long as they make sense in your life.",
            options: const [
              'ABANDONMENT',
              'AFFECTED',
              'AFFECTION',
              'INCREASE',
              'ANTIPATHY',
              'APATHY',
              'ATTACHMENT',
              'APPREHENSION',
              'GASP',
              'ATONY',
              'AUTHENTICITY',
              'SELF-KILLING',
              'SELF-ESTEEM',
              'BEAUTY',
              'KINDNESS',
              'COMPLACENCY',
              'COMPULSION',
              'COURAGE',
              'DECENCY',
              'DISAFFECTION',
              'DISCOURAGEMENT',
              'DISHARMONY',
              'FEARLESSNESS',
              'DIGNITY',
              'ELEGANCE',
              'EMPATHY',
              'DECEPTION',
              'HOPE',
              'ESTEEM',
              'NARROWNESS',
              'EUPHORIA',
              'FORTUNE',
              'FAILURE',
              'WEAKNESS',
              'FLEETING',
              'KINDNESS',
              'GRATITUDE',
              'HARMONY',
              'HUMILITY',
              'IMPASSIVENESS',
              'IMPOTENCE',
              'IMPULSIVITY',
              'LOSS OF APPETITE',
              'INDIFFERENCE',
              'INDULGENCE',
              'RESTLESSNESS',
              'INTELLIGENCE',
              'LETHARGY',
              'HURT',
              'MANIA',
              'MACHIAVELLIAN',
              'MELANCHOLY',
              'DEATH',
              'NOSTALGIA',
              'OBSESSION',
              'OBSESSION (INTENSE)',
              'PERISHABLE',
              'PERSISTENCE',
              'WORRY',
              'PROSTRATION',
              'PRUDENCE',
              'RUMINATION',
              'ANGER',
              'RESENTMENT',
              'SATISFACTION',
              'SECRECY',
              'LONELINESS',
              'SUICIDE',
              'TENACITY',
              'VIRTUE',
            ],
            optionsColumnsSize: 2,
            mimSizeAnswer: 1,
            maxSizeAnswer: 65,
          ),
        ],
  },
  67: {
    'hasProx': true,
    'header': 'Attention!!',
    'delay': 3,
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            "On the next screen, a list of 10 pairs of logically related words will be played (e.g., high–low). Then, you will be asked to fill in the missing word. Pay attention — you must memorize all pairs.\n\nWhen you're ready, click next...",
        bodyHasFrame: false,
      ),
    ],
  },
  68: {
    'hasProx': false,
    'header': 'Press play to listen!!',
    'answerLenght': 0,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          DisplayFrame(
            id: controller.idPage.value,
            body: 'assets/audios/dez_palavras.mp3',
            bodyHasFrame: true,
            playMusic: controller.playMusic,
          ),
        ],
  },
  69: {
    'hasProx': true,
    'header': 'Complete with the corresponding pair heard earlier',
    'answerLenght': 10,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TextFormList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: 'rain -',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: 'child -',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: '- summer',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[3]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: 'monster -',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[4]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: '- water',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[5]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: 'money -',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[6]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: '- small',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[7]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: 'book -',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[8]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: '- furniture',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          TextFormList(
            answer: controller.answerAux.value[9]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            optionsColumnsSize: 1,
            options: 'Teacher -',
            validator: (value) {
              if (value == null) return 'Incorrect data!!';
              if (value.isEmpty || value.length < 2) return 'Incorrect data!!';
              return null;
            },
            keyboardType: TextInputType.name,
            inputFormatters: [FilteringTextInputFormatter.singleLineFormatter],
          ),
        ],
  },
  70: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Do you worry about the internet (thinking about past virtual activities or anticipating your next connection)?',
              'Do you feel the need to use the internet for increasingly longer periods to feel satisfied?',
              'Have you repeatedly tried to control, reduce, or stop using the internet but failed?',
              'Do you become restless, moody, depressed, or irritable when trying to reduce or stop internet use, or when use is restricted?',
              'Do you stay online longer than you originally intended?',
              'Have you harmed or risked losing an important relationship, job, or educational opportunity because of the internet?',
              'Have you lied to family members, therapists, or others to hide the extent of your internet involvement?',
              'Do you use the internet as a way to escape problems or relieve unpleasant moods (e.g., helplessness, loneliness, guilt, sadness, anxiety, depression)?',
            ],
          ),
        ],
  },
  71: {
    'hasProx': true,
    'header': 'How often do you pause and consume these substances?',
    'answerLenght': 5,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: '1) Caffeine:',
            icon: Icons.more_time,
            options: const [
              'Never',
              'Once a month or less (rarely)',
              'Two to four times a month (sometimes)',
              'Two to three times a week',
              'Most days or always',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[1]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: '2) Alcohol:',
            icon: Icons.more_time,
            options: const [
              'Never',
              'Once a month or less (rarely)',
              'Two to four times a month (sometimes)',
              'Two to three times a week',
              'Most days or always',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[2]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: '3) Tobacco:',
            icon: Icons.more_time,
            options: const [
              'Never',
              'Once a month or less (rarely)',
              'Two to four times a month (sometimes)',
              'Two to three times a week',
              'Most days or always',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[3]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: '4) Marijuana:',
            icon: Icons.more_time,
            options: const [
              'Never',
              'Once a month or less (rarely)',
              'Two to four times a month (sometimes)',
              'Two to three times a week',
              'Most days or always',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
          const SizedBox(height: 10.0),
          const Divider(),
          const SizedBox(height: 10.0),

          SingleSelectionList(
            answer: controller.answerAux.value[4]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            title: '5) Cocaine/crack:',
            icon: Icons.more_time,
            options: const [
              'Never',
              'Once a month or less (rarely)',
              'Two to four times a month (sometimes)',
              'Two to three times a week',
              'Most days or always',
            ],
            optionsColumnsSize: 1,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  72: {
    'hasProx': true,
    'header': 'Answer the questions below:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          TypeYesNo(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            options: const [
              'Do you have intense fear or anxiety about a specific object or situation (e.g., flying, heights, animals, injections, blood)?',
              'Do you have persistent difficulty discarding or parting with possessions, regardless of their actual value?',
              'Do you experience strong fear or anxiety in one or more social situations where you may be exposed to possible scrutiny by others? Examples include social interactions (e.g., conversations, meeting unfamiliar people), being observed (eating or drinking), or performing in front of others (giving speeches, public speaking).',
              'Do you have no close friends or confidants who are not first-degree relatives, and generally seek leisure and/or work activities in solitude?',
              'Do you adopt an extremely low spending style for yourself and others, seeing money as something to be saved for potential future emergencies?',
              'Have you been experiencing feelings of worthlessness or excessive guilt?',
              'Have you started more projects than usual or engaged in riskier activities than you normally would?',
            ],
          ),
        ],
  },
  73: {
    'hasProx': true,
    'header': 'Choose!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(() {
                if (controller.answerAux.value[0].value == 'I am colorblind') {
                  controller.idPage.value = controller.idPage.value + 2;
                }
                state.currentState!.didChange(controller.answerAux.value);
              }),
            title:
                'Colorblindness is the term used to describe the lack of ocular sensitivity some people have in perceiving certain colors. Are you colorblind? Were you diagnosed by a specialized professional? If yes, click "I am colorblind." If not, click "I am not colorblind" and continue the activity.\n',
            icon: Icons.more_time,
            options: const ['I am colorblind', 'I am not colorblind'],
            optionsColumnsSize: 3,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  74: {
    'hasProx': true,
    'header': 'Let’s color!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(
            body:
                'Choose the color you like the most and paint the circles in the image below. Do it in any way you prefer.\n',
            bodyHasFrame: false,
          ),
          ArvoreCiculos(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            imagem: 'assets/arvore_circulos.png',
            itens: const [
              'Blue',
              'Green',
              'Orange',
              'Yellow',
              'Red',
              'Pink',
              'Brown',
              'Black',
              'Grey',
            ],
            optionsColumnsSize: 3,
            colors: const [
              Colors.blue,
              Colors.green,
              Colors.orange,
              Colors.yellow,
              Colors.red,
              Colors.pink,
              Colors.brown,
              Colors.black,
              Colors.grey,
            ],
          ),
        ],
  },
  75: {
    'hasProx': true,
    'header': 'Let’s color!!',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          const DisplayFrame(
            body:
                'Choose the color you like the most and paint the rectangles in the image below. Do it in any way you prefer.\n',
            bodyHasFrame: false,
          ),
          MuroColorido(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            imagem: 'assets/arvore_circulos.png',
            itens: const [
              'Blue',
              'Green',
              'Orange',
              'Yellow',
              'Red',
              'Pink',
              'Brown',
              'Black',
              'Grey',
            ],
            optionsColumnsSize: 3,
            colors: const [
              Colors.blue,
              Colors.green,
              Colors.orange,
              Colors.yellow,
              Colors.red,
              Colors.pink,
              Colors.brown,
              Colors.black,
              Colors.grey,
            ],
          ),
        ],
  },
  76: {
    'hasProx': true,
    'header': 'Select which day of the week is today:',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            options: const [
              'Monday',
              'Tuesday',
              'Wednesday',
              'Thursday',
              'Friday',
              'Saturday',
              'Sunday',
            ],
            optionsColumnsSize: 2,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  77: {
    'hasProx': true,
    'header':
        'How much time, approximately, do you think you spent completing this test?',
    'answerLenght': 1,
    'itens':
        (
          TelasController controller,
          GlobalKey<FormFieldState<List<ValueNotifier<String>>>> state,
        ) => [
          SingleSelectionList(
            answer: controller.answerAux.value[0]
              ..addListener(
                () => state.currentState!.didChange(controller.answerAux.value),
              ),
            icon: Icons.more_time,
            options: const [
              '5 minutes',
              '15 minutes',
              '30 minutes',
              '40 minutes',
              '60 minutes',
              'More than 1 hour',
            ],
            optionsColumnsSize: 2,
            hasPrefiroNaoDizer: false,
          ),
        ],
  },
  78: {
    'hasProx': false,
    'header': 'Congratulations!!!!',
    'answerLenght': 0,
    'itens': (_, __) => [
      const DisplayFrame(
        body:
            "You have completed the questionnaire. We greatly appreciate your time and availability.\n\nPS: You may now close this page.",
        bodyHasFrame: false,
      ),
    ],
  },
};
